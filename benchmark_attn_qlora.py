#!/usr/bin/env python
"""
benchmark_attn_qlora_v2.py
Finetune the same model with LoRA/QLoRA under different attention
implementations and report speed / memory / profiling breakdown & broader evaluation metrics.

USAGE
-----
python benchmark_attn_qlora_v2.py deepseek-ai/deepseek-coder-1.3b-base configs/comparison.json runs/bench
"""
import argparse, os, json, time, gc, math, random, csv
from typing import Dict, List

import torch
import torch.nn
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
import evaluate
from bitsandbytes.nn import Linear4bit

# custom attention implementations
from attentions.pa import PagedAttention
from attentions.mla import MultiHeadLatentAttention


train_size = 10
val_size = 5

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("cfg_file")
    p.add_argument("out_dir")
    return p.parse_args()

def get_attention_class(model: torch.nn.Module) -> type:
    """Find the first Attention class in a model to use as replacement template."""
    for module in model.modules():
        if module.__class__.__name__.endswith("Attention"):
            return module.__class__
    raise ValueError("Could not find an Attention class in the model")

def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

def unload_model(model: torch.nn.Module) -> None:
    """Properly unload model without moving quantized layers to CPU"""
    del model
    torch.cuda.empty_cache()
    gc.collect()


def greedy_gen_tokens_per_s(
    model: torch.nn.Module,
    prompt: str = "Benchmark",
    max_new: int = 32,
    n_trials: int = 3,
) -> float:
    """Measure generation throughput (tokens/sec) with greedy decode."""
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)

    torch.cuda.synchronize()
    total_tokens, total_time = 0, 0.0
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.time()
        _ = model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
            do_sample=False,
        )
        torch.cuda.synchronize()
        total_time += time.time() - start
        total_tokens += max_new
    return round(total_tokens / total_time, 2)

def replace_attention_with_quantized(
    orig_attn: torch.nn.Module,
    impl: str,
    model_config,
) -> torch.nn.Module:
    """
    Swap in either PagedAttention or MultiHeadLatentAttention,
    re‑using the original quantized q/k/v/o projections only.
    """
    if impl == "mla":
        custom_attn = MultiHeadLatentAttention(model_config)

    elif impl == "paged":
        custom_attn = PagedAttention(model_config, block_size=64)

    else:
        raise ValueError(f"Unsupported attention: {impl}")

    return custom_attn

def main():
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
    global tok
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.pad_token = tok.pad_token or tok.eos_token

    train_ds = load_dataset("Open-Orca/OpenOrca", split=f"train[:{train_size}]")
    val_ds   = load_dataset("Open-Orca/OpenOrca", split=f"train[{train_size}:{train_size+val_size}]")
    raw_val_ds = val_ds  # keep raw for generation

    max_len = cfg.get("max_seq_len", tok.model_max_length)
    def tokenize_fn(ex):
        instr = ((ex.get("system_prompt") + "\n") if ex.get("system_prompt") else "") + ex["question"]
        text = instr.strip() + "\n" + ex["response"].strip()
        return tok(text, truncation=True, max_length=max_len)

    train_set = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    val_set   = val_ds.map(tokenize_fn,   remove_columns=val_ds.column_names)

    print(train_set)
    print(val_set)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=False, pad_to_multiple_of=8
    )

    rouge = evaluate.load("rouge")

    ATTN_IMPLS: List[str] = cfg.get("attn_impls", [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[Dict] = []
    for impl in ATTN_IMPLS:
        run_dir = os.path.join(args.out_dir, impl)
        os.makedirs(run_dir, exist_ok=True)

        # Load model with built-in or custom attention
        if impl in ["eager", "sdpa", "flash_attention_2", "flex_attention"]:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                attn_implementation=impl,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                attn_implementation="eager",
            )
            for layer in model.model.layers:
                orig_attn = layer.self_attn
                custom_attn = replace_attention_with_quantized(
                    orig_attn,
                    impl,
                    model.config,
                )
                # move to the right device and install
                layer.self_attn = custom_attn.to(orig_attn.q_proj.weight.device)

        # Prepare for QLoRA
        model = prepare_model_for_kbit_training(model)
        model.enable_input_require_grads()

        # Verify attention modules by class type
        for idx, layer in enumerate(model.model.layers):
            attn_mod = layer.self_attn
            if impl == "paged" and not isinstance(attn_mod, PagedAttention):
                raise RuntimeError(f"Layer {idx} expected PagedAttention, got {attn_mod.__class__.__name__}")
            elif impl == "mla" and not isinstance(attn_mod, MultiHeadLatentAttention):
                raise RuntimeError(f"Layer {idx} expected MultiHeadLatentAttention, got {attn_mod.__class__.__name__}")

        # Apply LoRA adapters
        if impl == "mla":
            target_modules = [
                "to_q_latent", "to_k_token", "to_v_token", "out_latent",
                "to_q_token", "to_k_latent", "to_v_latent", "out_token",
            ]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        try:
            model.gradient_checkpointing_enable(use_reentrant=False)
        except TypeError:
            model.gradient_checkpointing_enable()

        # Training setup
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
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

        # --- Profiling: single step ---
        model.train()
        batch = next(iter(trainer.get_train_dataloader()))
        for k in batch: batch[k] = batch[k].to(device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with record_function("train_step_profiling"):
                outputs = model(**batch)
                outputs.loss.backward()
        prof.export_chrome_trace(os.path.join(run_dir, "profiling_trace.json"))
        breakdown = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        with open(os.path.join(run_dir, "profiling_summary.txt"), "w") as pf:
            pf.write(breakdown)
        torch.cuda.empty_cache()

        # --- Full train & eval ---
        t0 = time.time()
        train_out = trainer.train()
        train_s   = time.time() - t0
        peak_mem  = torch.cuda.max_memory_allocated() / 2**20
        eval_out  = trainer.evaluate()
        gen_tok_s = greedy_gen_tokens_per_s(model)

        # --- Broader eval: ROUGE on up to 100 samples ---
        max_eval = min(100, len(raw_val_ds))
        samples = raw_val_ds.select(list(range(max_eval)))
        predictions, references = [], []
        model.eval()
        for ex in samples:
            enc = tok(
                ex["question"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len,
            )
            inp  = enc.input_ids.to(device)
            mask = enc.attention_mask.to(device)
            out = model.generate(
                inp,
                attention_mask=mask,
                max_new_tokens=32,
                pad_token_id=tok.pad_token_id,
            )
            pred = tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True)
            predictions.append(pred)
            references.append(ex["response"])
        rouge_scores = rouge.compute(predictions=predictions, references=references)

        # Record results
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
            "rouge1": round(rouge_scores["rouge1"], 4),
            "rouge2": round(rouge_scores["rouge2"], 4),
            "rougeL": round(rouge_scores["rougeL"], 4),
            "trainable_params_M": round(trainable / 1e6, 1),
            "total_params_M": round(total / 1e6, 1),
        })

        unload_model(model)

    # Save all results
    with open(os.path.join(args.out_dir, "bench_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out_dir, "bench_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("=== BENCHMARK SUMMARY ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()