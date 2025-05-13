"""
benchmark_attn_adapters.py (with Weights & Biases integration + EM / token‑F1 evaluation)
===============================================================================
Compare fine‑tuning *speed*, *peak memory*, **exact‑match accuracy**, and **token‑level F1** for

* **baseline** (no adapters, full‑precision)
* **LoRA**      (fp16 weights + LoRA)
* **QLoRA**     (4‑bit weights + LoRA)

across multiple self‑attention implementations.

Only training‑throughput metrics are reported so runs stay fast.

USAGE
-----
python benchmark_attn_adapters_wandb_accuracy.py deepseek-ai/deepseek-coder-1.3b-base \
     configs/test.json runs/bench
"""
from __future__ import annotations

import argparse, os, json, time, gc, random
from typing import Dict, List, Any

import numpy as np
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

import wandb  # WandB integration

# from attentions.pa import PagedAttention
from attentions.mla import MultiHeadLatentAttention

TRAIN_SIZE = 5     # samples for throughput benchmark
EVAL_SIZE  = 5     # separate samples for accuracy evaluation
SEED       = 42


# --------------------------------------
# Metrics: Exact‑Match and token‑level F1
# --------------------------------------

def _compute_precision_recall_f1(tp: int, fp: int, fn: int):
    """Utility for micro‑averaged P / R / F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_metrics(eval_pred):
    """Return **exact‑match (em)** and **token‑level F1**.

    * Masked positions where `labels == -100` are ignored.
    * EM counts a prediction as correct only when *all* non‑ignored tokens match.
    * Token‑F1 is micro‑averaged across all non‑ignored tokens.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask  = labels != -100  # shape: (B, L)

    # --------- Exact‑Match (sequence level) ---------
    # True if every token matches for the sequence (ignoring mask)
    matches = ((preds == labels) | (~mask)).all(axis=1)
    em = matches.mean() if matches.size > 0 else 0.0

    # ---------------- Token‑level F1 ----------------
    tp = int(((preds == labels) & mask).sum())
    fp = int(((preds != labels) & mask).sum())  # predictions exist wherever mask is True
    fn = fp  # since there is exactly one prediction per label position when masked;
             # mismatched tokens count as both FP and FN.
    _p, _r, f1 = _compute_precision_recall_f1(tp, fp, fn)

    return {"em": float(em), "token_f1": float(f1)}


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
    raise ValueError(impl)


def main():
    args = parse_args()
    cfg: Dict[str, Any] = json.load(open(args.cfg_file))

    # Create output folder early so WandB run can point to it
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(SEED); torch.manual_seed(SEED)

    # ---------------------------------------------------------
    # Initialise Weights & Biases (one run logs all experiments)
    # ---------------------------------------------------------
    wandb_run = wandb.init(
        project=cfg.get("wandb_project", "attn_bench"),
        name=os.path.basename(args.out_dir),
        config={
            "model_id": args.model_id,
            "train_size": TRAIN_SIZE,
            "eval_size": EVAL_SIZE,
            **cfg,
        },
        dir=args.out_dir,
        job_type="benchmark",
        reinit=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # -------------------------------------------------
    # Dataset: split small TRAIN and EVAL subsets
    # -------------------------------------------------
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    max_len = cfg.get("max_seq_len", tok.model_max_length)

    def tok_fn(ex):
        text = f"{ex.get('system_prompt','')}\n{ex['question']}\n{ex['response']}".strip()
        return tok(text, truncation=True, max_length=max_len)

    train_set = ds.select(range(TRAIN_SIZE)).map(tok_fn, remove_columns=ds.column_names)
    eval_set  = ds.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)).map(tok_fn, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

    attn_impls: List[str] = cfg.get("attn_impls", [])
    adapters: List[str]  = cfg.get("adapter_modes", ["lora", "qlora"])

    results: List[Dict[str, Any]] = []

    for adapter in adapters:
        for impl in attn_impls:
            tag = f"{adapter}_{impl}"
            run_dir = os.path.join(args.out_dir, tag)
            os.makedirs(run_dir, exist_ok=True)

            # ---------------------------
            # Model & adapter setup block
            # ---------------------------
            if adapter == "qlora":
                # QLoRA: 4-bit
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id, quantization_config=bnb_cfg, device_map="auto"
                )
                model = prepare_model_for_kbit_training(model)
            elif adapter == "lora":
                # LoRA
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            else:  # baseline
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )

            torch.cuda.reset_peak_memory_stats()

            # Replace attention implementation if necessary
            if impl not in {"eager", "sdpa", "flash_attention_2", "flex_attention"}:
                for layer in model.model.layers:
                    ref = layer.self_attn.q_proj.weight
                    new_attn = custom_attention(impl, model.config)
                    target_dtype = ref.dtype if ref.dtype.is_floating_point else torch.float16
                    new_attn = new_attn.to(device=ref.device, dtype=target_dtype)
                    layer.self_attn = new_attn
            else:
                model.config.attn_implementation = impl

            # Attach (Q)LoRA adapters
            if adapter in {"lora", "qlora"}:
                target_modules = [
                    "to_q_latent","to_k_token","to_v_token","out_latent",
                    "to_q_token","to_k_latent","to_v_latent","out_token",
                ] if impl == "mla" else ["q_proj","k_proj","v_proj","o_proj"]
                lora_cfg = LoraConfig(
                    r=32, lora_alpha=16, lora_dropout=0.05, bias="none",
                    target_modules=target_modules, task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_cfg)
                model.enable_input_require_grads()

            model.gradient_checkpointing_enable()
            model.config.use_cache = False

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=run_dir,
                    run_name=tag,            # Shows up in WandB timeline
                    seed=SEED,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=32,
                    num_train_epochs=1,
                    logging_steps=10,
                    save_total_limit=1,
                    report_to="wandb",     # Activate WandB callback
                    evaluation_strategy="no",
                    gradient_checkpointing=True,
                    fp16=False,
                    optim="paged_adamw_32bit",
                ),
                train_dataset=train_set,
                eval_dataset=eval_set,
                data_collator=collator,
                compute_metrics=compute_metrics,
            )

            # ---------------------
            # Profiling + training
            # ---------------------
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                with record_function("train_with_profiler"):
                    start = time.time()
                    train_out = trainer.train()
                    runtime = time.time() - start
                    prof.step()

            # ---------------------
            # Evaluation metrics
            # ---------------------
            eval_metrics = trainer.evaluate(eval_dataset=eval_set)
            em  = round(eval_metrics.get("eval_em", 0.0), 4)
            f1  = round(eval_metrics.get("eval_token_f1", 0.0), 4)

            peak_mem = (
                torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
            )

            result = {
                "mode": adapter,
                "impl": impl,
                "train_runtime_s": round(runtime, 1),
                "train_samples_per_s": round(train_out.metrics.get("train_samples_per_second", 0), 3),
                "peak_mem_MiB": int(peak_mem),
                "eval_em": em,
                "eval_token_f1": f1,
            }
            results.append(result)

            # Log to WandB immediately for real‑time dashboard
            wandb.log({f"metrics/{tag}": result}, step=len(results))

            unload(model)

    # -----------
    # Final dump
    # -----------
    summary_path = os.path.join(args.out_dir, "bench_results.json")
    json.dump(results, open(summary_path, "w"), indent=2)
    wandb.save(summary_path)   # attach artefact to the run

    # Optional: log aggregated results table
    wandb.log({"results": wandb.Table(data=[list(r.values()) for r in results],
                                       columns=list(results[0].keys()))})

    print("\n== SPEED + EM / TOKEN‑F1 BENCHMARK ==")
    for r in results:
        print(r)

    wandb.finish()


if __name__ == "__main__":
    main()
