# src/5_train.py   🖥️ RUNS ON VAST      usage: python src/5_train.py A 0
# ===== EDITED BY CLAUDE 2026-07-18: production hardening + version fixes =====
# 1. save_strategy="no" (was "epoch"): don't write a checkpoint every epoch for all 9 runs —
#    that bloats Vast disk fast. We save ONE adapter at the end via trainer.save_model().
# 2. gradient_checkpointing=True + use_reentrant flag: trades a little speed for lower VRAM,
#    so a 4090 (24GB) comfortably fits Llama-3.2-3B + LoRA without OOM.
# 3. max_seq_length -> max_length: newer TRL renamed this SFTConfig argument.
# 4. removed assistant_only_loss=True: Llama-3.2's chat template isn't training-compatible with it;
#    model trains on full Q+A sequence instead (applied identically to A/B/C, so comparison holds).
# The experiment-critical parts (identical config across A/B/C, seeds) are UNCHANGED.
# ============================================================================
import json, sys
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

COND = sys.argv[1]       # "A" | "B" | "C"  -> which training file
SEED = int(sys.argv[2])  # 0 | 1 | 2        -> 3 seeds so you can report mean±std
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token

print(f"[{COND} seed {SEED}] loading training data...")
rows=[json.loads(l) for l in open(f"data/train_{COND}.jsonl")]
def to_chat(o):
    return {"messages":[{"role":"user","content":o["Q"]},
                        {"role":"assistant","content":o["A"]}]}
ds = Dataset.from_list([to_chat(o) for o in rows])
print(f"[{COND} seed {SEED}] {len(ds)} training rows loaded")

lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

cfg = SFTConfig(output_dir=f"out/{COND}_s{SEED}",
    per_device_train_batch_size=8, gradient_accumulation_steps=4,
    learning_rate=2e-5, lr_scheduler_type="constant", num_train_epochs=3,
    logging_steps=10, seed=SEED, bf16=True,           # loss printed every 10 steps
    max_length=512, packing=False,                    # EDITED 2026-07-18: max_seq_length -> max_length (newer TRL)
    save_strategy="no",                               # EDITED: no per-epoch checkpoints (disk bloat)
    gradient_checkpointing=True,                      # EDITED: lower VRAM, fits 24GB comfortably
    gradient_checkpointing_kwargs={"use_reentrant": False})
    # EDITED 2026-07-18: removed assistant_only_loss=True — Llama-3.2's chat template lacks the
    # {% generation %} markers TRL needs, so it can't mask question tokens. Model now trains on the
    # full Q+A sequence. This is applied IDENTICALLY to A/B/C, so the controlled comparison is intact;
    # note "trained on full QA sequence" in the paper's method section.

print(f"[{COND} seed {SEED}] loading base model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False                        # EDITED: required when gradient_checkpointing=True
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
print(f"[{COND} seed {SEED}] training started (loss prints every 10 steps)...")
trainer.train()
trainer.save_model(f"out/{COND}_s{SEED}/adapter")   # saves the LoRA adapter cleanly (PEFT-aware)
print(f"✅ [{COND} seed {SEED}] done — adapter saved to out/{COND}_s{SEED}/adapter")