#!/usr/bin/env python
"""
benchmark_attn_adapters.py
=========================
Compare fine‑tuning *speed* (and peak memory) for

* **baseline** (no adapters, full‑precision)
* **LoRA**      (fp16 weights + LoRA)
* **QLoRA**     (4‑bit weights + LoRA)

across multiple self‑attention implementations.

Only training‑throughput metrics are reported so runs stay fast.

USAGE
-----
python benchmark_attn_adapters.py deepseek-ai/deepseek-coder-1.3b-base \
     configs/test.json runs/bench
"""
from __future__ import annotations

import argparse, os, json, time, gc, random
from typing import Dict, List, Any

import torch
from torch.profiler import profile, record_function, ProfilerActivity
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

# from attentions.pa import PagedAttention
from attentions.mla import MultiHeadLatentAttention

TRAIN_SIZE = 100
SEED       = 42

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("cfg_file")
    p.add_argument("out_dir")
    return p.parse_args()


def unload(model: torch.nn.Module):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def custom_attention(impl: str, cfg):
    if impl == "mla":
        return MultiHeadLatentAttention(cfg, num_latents=64, dropout=cfg.attention_dropout)
    if impl == "paged":
        return PagedAttention(cfg, block_size=64)
    raise ValueError(impl)

def main():
    args = parse_args()
    cfg: Dict[str, Any] = json.load(open(args.cfg_file))

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(SEED); torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.pad_token = tok.eos_token

    ds = load_dataset("Open-Orca/OpenOrca", split=f"train[:{TRAIN_SIZE}]")
    max_len = cfg.get("max_seq_len", tok.model_max_length)

    def tok_fn(ex):
        text = f"{ex.get('system_prompt','')}\n{ex['question']}\n{ex['response']}".strip()
        return tok(text, truncation=True, max_length=max_len)

    train_set = ds.map(tok_fn, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

    attn_impls: List[str] = cfg.get("attn_impls", [])
    adapters: List[str]  = cfg.get("adapter_modes", ["baseline", "lora", "qlora"])

    results: List[Dict[str, Any]] = []

    for adapter in adapters:
        for impl in attn_impls:
            tag = f"{adapter}_{impl}"
            run_dir = os.path.join(args.out_dir, tag)
            os.makedirs(run_dir, exist_ok=True)

            if adapter == "qlora":
                bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_cfg, device_map="auto")
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)

            # swap attention if custom
            if impl not in {"eager", "sdpa", "flash_attention_2", "flex_attention"}:
                for layer in model.model.layers:
                    ref = layer.self_attn.q_proj.weight          # reference tensor
                    new_attn = custom_attention(impl, model.config)

                    # choose a safe fp dtype: keep ref.dtype if it is already fp;
                    # otherwise fall back to fp16 (or bf16 if you prefer)
                    target_dtype = (
                        ref.dtype if ref.dtype.is_floating_point else torch.float16
                    )

                    new_attn = new_attn.to(device=ref.device, dtype=target_dtype)
                    layer.self_attn = new_attn
            else:
                model.config.attn_implementation = impl

            if adapter in {"lora", "qlora"}:
                target_modules = [
                    "to_q_latent","to_k_token","to_v_token","out_latent",
                    "to_q_token","to_k_latent","to_v_latent","out_token",
                ] if impl == "mla" else ["q_proj","k_proj","v_proj","o_proj"]
                lora_cfg = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=target_modules, task_type="CAUSAL_LM")
                if adapter == "qlora":
                    model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, lora_cfg)
                model.enable_input_require_grads()

            model.gradient_checkpointing_enable()

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=run_dir,
                    seed=SEED,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=32,
                    num_train_epochs=1,
                    logging_steps=10,
                    save_total_limit=1,
                    report_to="none",
                    #evaluation_strategy="no",
                    gradient_checkpointing=True,
                    fp16=False,
                ),
                train_dataset=train_set,
                data_collator=collator,
            )

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir),
                record_shapes=True,
                with_stack=True,
                with_flops=True
            ) as prof:
                with record_function("train_with_profiler"):
                    start = time.time()
                    train_metrics = trainer.train()
                    runtime = time.time() - start
                    prof.step()

            peak_mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0

            results.append({
                "mode": adapter,
                "impl": impl,
                "train_runtime_s": round(runtime, 1),
                "train_samples_per_s": round(train_metrics.metrics.get("train_samples_per_second", 0), 3),
                "peak_mem_MiB": int(peak_mem),
            })

            unload(model)

    json.dump(results, open(os.path.join(args.out_dir,"bench_results.json"),"w"), indent=2)
    print("\n== SPEED BENCHMARK ==")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
