# src/5_train.py   🖥️ RUNS ON VAST      usage: python src/5_train.py A 0
# ===== EDITED BY CLAUDE 2026-07-21: save adapters to HuggingFace so they survive instance death =====
# Adds an automatic push of each finished adapter to a PRIVATE HF repo. The previous run's adapters
# were lost when the Vast instance was destroyed, which forced a full retrain to add direct-recall.
# Set HF_USER below. Requires: `hf auth login` (write token) on the box before running.
# All experiment-critical config (identical across A/B/C, seeds, r=64) is UNCHANGED.
# ==================================================================================================
import json, sys, os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

COND = sys.argv[1]       # "A" | "B" | "C"
SEED = int(sys.argv[2])  # 0 | 1 | 2
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_USER = "Faisal2319"                              # <-- your HF username
PUSH = os.environ.get("RB_PUSH_ADAPTERS", "1") == "1"   # set RB_PUSH_ADAPTERS=0 to disable

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
    logging_steps=10, seed=SEED, bf16=True,
    max_length=512, packing=False,
    save_strategy="no",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})

print(f"[{COND} seed {SEED}] loading base model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
print(f"[{COND} seed {SEED}] training started (loss prints every 10 steps)...")
trainer.train()

adapter_dir = f"out/{COND}_s{SEED}/adapter"
trainer.save_model(adapter_dir)
print(f"✅ [{COND} seed {SEED}] adapter saved locally to {adapter_dir}")

# ===== push to a private HF repo so it is never lost again =====
if PUSH:
    try:
        from huggingface_hub import HfApi
        repo_id = f"{HF_USER}/ripplebudget-adapters"
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=adapter_dir, repo_id=repo_id,
                          path_in_repo=f"{COND}_s{SEED}", repo_type="model")
        print(f"✅ [{COND} seed {SEED}] pushed to hf.co/{repo_id}/{COND}_s{SEED}")
    except Exception as e:
        print(f"⚠️  push failed for {COND} seed {SEED}: {e}\n   (adapter is still saved locally at {adapter_dir})")