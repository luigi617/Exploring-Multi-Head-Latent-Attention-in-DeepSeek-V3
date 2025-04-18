import sys, torch, json, os
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)          # ⚙ A
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from torch_profiling import TorchProfilerCallback
# ------------------------------------------------------------------
#  load cfg / model / tokenizer
# ------------------------------------------------------------------
model_id, cfg_file, out_dir = sys.argv[1:4]
cfg = json.load(open(cfg_file))

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# ⚙ B — make sure the tokenizer *has* a pad token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token        # safe default for causal‑LMs

# ------------------------------------------------------------------
#  pick the attention implementation (flash, paged, mla, …)
# ------------------------------------------------------------------
def patch(model, impl):
    for layer in model.model.layers:
        layer.self_attn.config.attn_impl = impl
patch_attn = cfg["attn_impl"]

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_cfg,
                                             device_map="auto")
patch(model, patch_attn)

lora_cfg = LoraConfig(r=32, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_cfg)

# ------------------------------------------------------------------
#  dataset (leave *padding* to the collator)
# ------------------------------------------------------------------
ds = load_dataset("Open-Orca/OpenOrca", split="train[:5000]")
max_len = cfg.get("max_seq_len", model.config.max_position_embeddings)

def fmt(ex):
    # build a simple instruction‑follow prompt
    instr = (ex["system_prompt"] + "\n" if ex["system_prompt"] else "") + ex["question"]
    text = instr.strip() + "\n" + ex["response"].strip()
    return tok(text, truncation=True, max_length=max_len)

train_set = ds.map(fmt, remove_columns=ds.column_names)

# ⚙ C — dynamic‑padding collator (copies input_ids → labels; masks pads with −100)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tok,
    mlm=False,                     # causal‑LM
    pad_to_multiple_of=8           # helps tensor cores, optional
)

# ------------------------------------------------------------------
#  training
# ------------------------------------------------------------------


args = TrainingArguments(
    output_dir=out_dir,
    bf16=False,                              # we use pure fp16
    fp16=True,
    per_device_train_batch_size=1,           # 🢂 tiny micro‑batch
    gradient_accumulation_steps=32,          # global batch 32
    num_train_epochs=1,
    logging_steps=10,
    save_total_limit=1,
)

Trainer(model=model,
        args=args,
        train_dataset=train_set,
        data_collator=data_collator,
        callbacks=[TorchProfilerCallback(wait=2, warmup=2, active=8)]
        ).train()
