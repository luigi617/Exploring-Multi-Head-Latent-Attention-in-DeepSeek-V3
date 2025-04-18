#!/usr/bin/env python
"""
benchmark_attn_qlora.py
Finetune the same model with LoRA/QLoRA under different attention
implementations and report speed / memory / quality metrics.

USAGE
-----
python benchmark_attn_qlora.py deepseek-ai/deepseek-coder-1.3b-base configs/comparison.json runs/bench
"""
from __future__ import annotations
import argparse, os, json, time, gc, math, random
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from bitsandbytes.nn import Linear4bit

# custom attention implementations
from attentions.naive import NaiveAttention
from attentions.MHA import MultiHeadAttention
from attentions.PA import PagedAttention

# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("cfg_file")
    p.add_argument("out_dir")
    return p.parse_args()

def replace_linear(
    model: torch.nn.Module,
    lin_cls: type[torch.nn.Module],
    skip_modules: list[str] | None = None,
    **bnb_kwargs,
) -> None:
    if skip_modules is None:
        skip_modules = ["lm_head"]
    for name, child in list(model.named_children()):
        if name in skip_modules:
            continue
        replace_linear(child, lin_cls, skip_modules=skip_modules, **bnb_kwargs)
        if isinstance(child, torch.nn.Linear):
            model._modules[name] = lin_cls(
                child.in_features,
                child.out_features,
                child.bias is not None,
                **bnb_kwargs,
            )

def get_attention_class(model: torch.nn.Module) -> type:
    # find any class with "Attention" in its name
    for module in model.modules():
        if module.__class__.__name__.endswith("Attention"):
            return module.__class__
    raise ValueError("Could not find an Attention class in the model")

def replace_attention_layers(
    model: torch.nn.Module,
    original_cls: type,
    new_cls: type,
    load_state: bool = True,
) -> None:
    for name, module in list(model.named_children()):
        if isinstance(module, original_cls):
            cfg = module.config
            device = next(module.parameters()).device
            new_mod = new_cls(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
            ).to(device)
            if load_state:
                new_mod.load_state_dict(module.state_dict(), strict=False)
            model._modules[name] = new_mod
        else:
            replace_attention_layers(module, original_cls, new_cls, load_state)

def patch_attn(model: torch.nn.Module, impl: str) -> None:
    for layer in model.model.layers:
        layer.self_attn.config.attn_impl = impl

def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

def unload_model(model: torch.nn.Module) -> None:
    model.to("cpu", dtype=torch.float32)
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

@torch.inference_mode()
def greedy_gen_tokens_per_s(
    model: torch.nn.Module,
    prompt: str = "Benchmark",
    max_new: int = 32,
    n_trials: int = 3,
) -> float:
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    torch.cuda.synchronize()
    total_tokens, total_time = 0, 0.0
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.time()
        _ = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        total_time += time.time() - start
        total_tokens += max_new
    return round(total_tokens / total_time, 2)

# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
args = parse_args()
cfg = json.load(open(args.cfg_file))
os.makedirs(args.out_dir, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# quantization config
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# tokenizer & datasets
tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
tok.pad_token = tok.pad_token or tok.eos_token

train_ds = load_dataset("Open-Orca/OpenOrca", split="train[:500]")
val_ds = load_dataset("Open-Orca/OpenOrca", split="train[500:600]")

max_len = cfg.get("max_seq_len", tok.model_max_length)
def tokenize_fn(ex):
    instr = ((ex["system_prompt"] + "\n") if ex["system_prompt"] else "") + ex["question"]
    text = instr.strip() + "\n" + ex["response"].strip()
    return tok(text, truncation=True, max_length=max_len)

train_set = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
val_set   = val_ds.map(tokenize_fn,   remove_columns=val_ds.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tok, mlm=False, pad_to_multiple_of=8
)

ATTN_IMPLS: List[str] = cfg.get("attn_impls", [])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results: List[Dict] = []
for impl in ATTN_IMPLS:
    run_dir = os.path.join(args.out_dir, impl)
    os.makedirs(run_dir, exist_ok=True)

    # load with built‑in attention backends
    if impl in ["eager", "sdpa", "flash_attention_2", "flash_attention_3"]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation=impl,
        )
    else:
        # start from eager and then swap in custom layers
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation="eager",
        )
        orig_cls = get_attention_class(model)

        if impl == "naive":
            # preserve original weights
            orig_w = {
                f"layer_{i}": layer.self_attn.state_dict()
                for i, layer in enumerate(model.model.layers)
            }
            replace_attention_layers(model, orig_cls, NaiveAttention, load_state=False)
            replace_linear(
                model,
                Linear4bit,
                compute_dtype=torch.float16,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )
            # reload weights into 4‑bit modules
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.load_state_dict(orig_w[f"layer_{i}"], strict=False)

        elif impl == "paged":
            replace_attention_layers(model, orig_cls, PagedAttention)

        elif impl == "mha":
            replace_attention_layers(model, orig_cls, MultiHeadAttention)

        else:
            raise ValueError(f"Unsupported attention implementation: {impl}")

    # prepare for QLoRA
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    # verify patch
    patch_attn(model, impl)
    for idx, layer in enumerate(model.model.layers):
        if layer.self_attn.config.attn_impl != impl:
            raise RuntimeError(
                f"Layer {idx} attn_impl mismatch: "
                f"expected '{impl}', got '{layer.self_attn.config.attn_impl}'"
            )

    # attach LoRA
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # gradient checkpointing fallback
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable()

    # count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    # trainer
    tr_args = TrainingArguments(
        output_dir=run_dir,
        fp16=True,
        seed=SEED,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
    )

    # train
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_out = trainer.train()
    train_s = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20

    # eval
    eval_out = trainer.evaluate()

    # generation micro‑benchmark
    gen_tok_s = greedy_gen_tokens_per_s(model)

    results.append({
        "impl": impl,
        "train_runtime_s": round(train_s, 1),
        "train_samples_per_s": round(train_out.metrics.get("train_samples_per_second", 0), 3),
        "eval_samples_per_s": round(eval_out.get("eval_samples_per_second", 0), 3),
        "gen_tok_per_s": gen_tok_s,
        "peak_mem_MiB": int(peak_mem),
        "train_loss": round(train_out.training_loss, 4),
        "val_loss": round(eval_out["eval_loss"], 4),
        "val_ppl": round(perplexity(eval_out["eval_loss"]), 2),
        "trainable_params_M": round(trainable / 1e6, 1),
        "total_params_M": round(total / 1e6, 1),
    })

    unload_model(model)

# save
with open(os.path.join(args.out_dir, "bench_results.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(args.out_dir, "bench_results.csv"), "w", newline="") as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("=== BENCHMARK SUMMARY ===")
for r in results:
    print(r)
print(f"\nSaved results to {args.out_dir}")
