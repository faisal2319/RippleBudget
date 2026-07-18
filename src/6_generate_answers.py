# src/6_generate_answers.py   🖥️ RUNS ON VAST
# ===== EDITED BY CLAUDE 2026-07-18: crash-proofing (this script lost 2h once to a late crash) =====
# 1. os.makedirs("results") — the folder isn't created by git clone (git skips empty dirs), which
#    caused a FileNotFoundError at the FINAL write step, losing the whole run.
# 2. INCREMENTAL writing — each answer is streamed to disk immediately (open in append mode),
#    so a crash never loses more than the current line, and progress survives.
# 3. RESUME — on restart it skips (cond,seed,question) triples already in the file, so a re-run
#    continues instead of starting over.
# ==================================================================================================
import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

os.makedirs("results", exist_ok=True)              # FIX 1: ensure output folder exists
OUT = "results/answers.jsonl"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL)
ripple=[json.loads(l) for l in open("data/eval_ripple.jsonl")]
locality=[json.loads(l) for l in open("data/eval_locality.jsonl")]

# FIX 3: load already-completed items so a re-run resumes instead of restarting.
done_keys=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r=json.loads(l); done_keys.add((r["cond"],r["seed"],r["split"],r["Q"]))
        except: pass
print(f"resuming: {len(done_keys)} answers already on disk")

def load(cond,seed):
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="auto")
    return PeftModel.from_pretrained(m, f"out/{cond}_s{seed}/adapter")

def gen(model,q):
    # newer transformers returns a dict from apply_chat_template; unpack with **ids for generate.
    ids = tok.apply_chat_template([{"role":"user","content":q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=64, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

total_units = 3*3*(len(ripple)+len(locality))
done = len(done_keys)
# FIX 2: open in APPEND mode and flush each line — nothing is held only in memory.
with open(OUT,"a") as f:
    for cond in ["A","B","C"]:
        for seed in [0,1,2]:
            # skip loading this adapter entirely if all its questions are already done
            remaining=[(sp,o) for sp,items in [("ripple",ripple),("locality",locality)]
                       for o in items if (cond,seed,sp,o["Q"]) not in done_keys]
            if not remaining:
                print(f"skip {cond} seed {seed} (already complete)"); continue
            model=load(cond,seed)
            print(f"loaded {cond} seed {seed} — {len(remaining)} answers to generate...")
            for split,o in remaining:
                rec={"cond":cond,"seed":seed,"split":split,
                     "cls":o.get("cls",""),"fact_id":o.get("fact_id",""),
                     "Q":o["Q"],"gold":o.get("A",""),"pred":gen(model,o["Q"])}
                f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()   # write immediately
                done+=1
                if done % 50 == 0:
                    print(f"  {done}/{total_units} answers generated")
print("✅ done — answers in", OUT)