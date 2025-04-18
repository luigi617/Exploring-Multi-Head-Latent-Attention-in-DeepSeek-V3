# ── finetune_attn_qlora.py ─────────────────────────────────────────────────────
"""
Benchmark quality + wall‑clock cost for different attention back‑ends *with QLoRA*.
"""

import argparse, time, gc, torch, json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training
)
from accelerate import Accelerator


# ------------------------------ data -----------------------------------------
def make_ds(tokenizer, max_len):
    train = load_dataset("Open-Orca/OpenOrca", split="train[:500]")
    val   = load_dataset("Open-Orca/OpenOrca", split="train[500:550]")

    def _tok(e):
        prompt = ((e["system_prompt"] or "") + e["question"]).strip()
        text   = prompt + "\n" + e["response"].strip()
        return tokenizer(text, truncation=True, max_length=max_len)
    return (
        train.map(_tok, remove_columns=train.column_names),
        val.map(_tok,   remove_columns=val.column_names)
    )


# ------------------------------ one run --------------------------------------
def run(model_id, impl, args):
    acc = Accelerator(log_with="tensorboard",
                      project_dir=f"{args.out_dir}/{impl}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token

    # 1️⃣  load backbone in 4‑bit
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.attn_impl = impl

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 2️⃣  make sure every layer actually switches kernel
    for layer in model.model.layers:
        layer.self_attn.config.attn_impl = impl

    # 3️⃣  QLoRA prep (gradient‑checkpointing + paged AdamW)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4️⃣  attach LoRA adapters
    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # 5️⃣  data
    train_ds, val_ds = make_ds(tok, model.config.max_position_embeddings)

    # 6️⃣  trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{args.out_dir}/{impl}",
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_acc,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=True,
            lr_scheduler_type="cosine",
            warmup_steps=max(20, args.max_steps // 20),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=200,
            save_total_limit=1,
            report_to="tensorboard"
        ),
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # 7️⃣  run
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    ppl = torch.exp(torch.tensor(trainer.evaluate()["eval_loss"]))
    acc.print(f"{impl:15} | perplexity {ppl:6.2f} | wall‑time {elapsed/60:5.1f} min")

    # tidy‑up to free VRAM for next run
    acc.end_training()
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()


# ------------------------------ cli ------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--impls", nargs="+", required=True,
                    help="e.g. mha flash_attn_v3 paged_attn latent_attn")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-acc", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=500,
                    help="Short run for demo; raise for serious training")
    args = ap.parse_args()

    for impl in args.impls:
        run(args.model_id, impl, args)
# ───────────────────────────────────────────────────────────────────────────────
