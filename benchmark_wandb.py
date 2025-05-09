"""
benchmark_attn_qlora_v2.py

+ Weights & Biases experiment tracking.
-----
python benchmark_attn_qlora_v2.py deepseek-ai/deepseek-coder-1.3b-base \
     configs/comparison.json runs/bench
"""
from __future__ import annotations
import argparse, os, json, time, gc, math, random, csv
from typing import Dict, List, Any

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from transformers.integrations import WandbCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import evaluate
import wandb

# custom attention implementations
# from attentions.pa  import PagedAttention
from attentions.mla import MultiHeadLatentAttention


TRAIN_SIZE, VAL_SIZE, SEED = 10, 5, 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("cfg_file")
    p.add_argument("out_dir")
    return p.parse_args()


def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def unload_model(model: torch.nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def greedy_gen_tokens_per_s(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    prompt: str = "Benchmark",
    max_new: int = 32,
    n_trials: int = 3,
) -> float:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_tokens, total_time = 0, 0.0

    for _ in range(n_trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        _ = model.generate(
            **enc, max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id, do_sample=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.time() - start
        total_tokens += max_new

    return round(total_tokens / max(total_time, 1e-6), 2)


def replace_attention_with_quantized(
    orig_attn: torch.nn.Module,
    impl: str,
    cfg: Any,
) -> torch.nn.Module:
    if impl == "mla":
        custom_attn = MultiHeadLatentAttention(cfg)
    elif impl == "paged":
        custom_attn = PagedAttention(cfg, block_size=64)
        custom_attn.q_proj = orig_attn.q_proj
        custom_attn.k_proj = orig_attn.k_proj
        custom_attn.v_proj = orig_attn.v_proj
        custom_attn.o_proj = orig_attn.o_proj
    else:
        raise ValueError(f"Unsupported attention implementation: {impl}")
    return custom_attn



def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = json.load(open(args.cfg_file))
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.pad_token = tok.eos_token

    train_ds = load_dataset("Open-Orca/OpenOrca", split=f"train[:{TRAIN_SIZE}]")
    val_ds   = load_dataset("Open-Orca/OpenOrca",
                            split=f"train[{TRAIN_SIZE}:{TRAIN_SIZE+VAL_SIZE}]")
    raw_val_ds = val_ds

    max_len = cfg.get("max_seq_len", tok.model_max_length)

    def tokenize_fn(ex):
        instr = ((ex.get("system_prompt") + "\n") if ex.get("system_prompt") else "") + ex["question"]
        text = instr.strip() + "\n" + ex["response"].strip()
        return tok(text, truncation=True, max_length=max_len)

    train_set = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    val_set   = val_ds.map(tokenize_fn,   remove_columns=val_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok,
                                                    mlm=False, pad_to_multiple_of=8)
    rouge = evaluate.load("rouge")
    impls: List[str] = cfg.get("attn_impls", [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[Dict[str, Any]] = []

    for impl in impls:
        run_dir = os.path.join(args.out_dir, impl)
        os.makedirs(run_dir, exist_ok=True)

        wandb_run = wandb.init(
            project="finetuning_test",
            name=f"{impl}-{args.model_id.split('/')[-1]}",
            dir=run_dir,
            reinit=True,
            config={
                "model_id": args.model_id,
                "impl": impl,
                **cfg,
                "train_size": TRAIN_SIZE,
                "val_size": VAL_SIZE,
                "seed": SEED,
            },
        )

        base_kwargs = dict(
            pretrained_model_name_or_path=args.model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
        )

        if impl in {"eager", "sdpa", "flash_attention_2", "flex_attention"}:
            model = AutoModelForCausalLM.from_pretrained(
                **base_kwargs, attn_implementation=impl
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                **base_kwargs, attn_implementation="eager"
            )
            for layer in model.model.layers:
                layer.self_attn = replace_attention_with_quantized(
                    layer.self_attn, impl, model.config
                ).to(layer.self_attn.q_proj.weight.device)

        model = prepare_model_for_kbit_training(model)
        model.enable_input_require_grads()

        for idx, layer in enumerate(model.model.layers):
            if impl == "paged" and not isinstance(layer.self_attn, PagedAttention):
                raise RuntimeError(f"Layer {idx} not PagedAttention")
            if impl == "mla" and not isinstance(layer.self_attn, MultiHeadLatentAttention):
                raise RuntimeError(f"Layer {idx} not MultiHeadLatentAttention")

        target_modules = (
            ["to_q_latent", "to_k_token", "to_v_token", "out_latent",
             "to_q_token", "to_k_latent", "to_v_latent", "out_token"]
            if impl == "mla"
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        lora_cfg = LoraConfig(
            r=32, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=target_modules, task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)
        try:
            model.gradient_checkpointing_enable(use_reentrant=False)
        except TypeError:
            model.gradient_checkpointing_enable()

        tr_args = TrainingArguments(
            output_dir=run_dir, fp16=True, seed=SEED,
            per_device_train_batch_size=4, gradient_accumulation_steps=32,
            num_train_epochs=1, logging_steps=10, save_total_limit=1,
            report_to="none", gradient_checkpointing=True,
        )
        trainer = Trainer(
            model=model,
            args=tr_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator,
            callbacks=[WandbCallback()]
        )

        if torch.cuda.is_available():
            model.train()
            batch = next(iter(trainer.get_train_dataloader()))
            batch = {k: v.to(device) for k, v in batch.items()}

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, with_stack=True
            ) as prof:
                with record_function("train_step_profiling"):
                    loss = model(**batch).loss
                    loss.backward()

            trace_path = os.path.join(run_dir, "profiling_trace.json")
            prof.export_chrome_trace(trace_path)
            with open(os.path.join(run_dir, "profiling_summary.txt"), "w") as pf:
                pf.write(prof.key_averages().table(sort_by="cpu_time_total",
                                                   row_limit=10))
            wandb.save(trace_path)
            wandb.save(os.path.join(run_dir, "profiling_summary.txt"))  # ★
            torch.cuda.empty_cache()

        start_t = time.time()
        train_out = trainer.train()
        train_runtime = time.time() - start_t
        peak_mem = (torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available() else 0)

        eval_out = trainer.evaluate()
        gen_tok_s = greedy_gen_tokens_per_s(model, tok)

        max_eval = min(100, len(raw_val_ds))
        predictions, references = [], []
        model.eval()
        for ex in raw_val_ds.select(range(max_eval)):
            enc = tok(ex["question"], return_tensors="pt",
                      truncation=True, padding=True, max_length=max_len).to(device)
            out = model.generate(**enc, max_new_tokens=32,
                                 pad_token_id=tok.pad_token_id)
            pred = tok.decode(out[0][enc.input_ids.shape[-1]:],
                              skip_special_tokens=True)
            predictions.append(pred)
            references.append(ex["response"])
        rouge_scores = rouge.compute(predictions=predictions,
                                     references=references)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())

        summary = {
            "impl": impl,
            "train_runtime_s": round(train_runtime, 1),
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
        }
        results.append(summary)
        wandb.log(summary)

        unload_model(model)
        wandb_run.finish()

    with open(os.path.join(args.out_dir, "bench_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out_dir, "bench_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    wandb.save(os.path.join(args.out_dir, "bench_results.*"))

    print("\n=== BENCHMARK SUMMARY ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()