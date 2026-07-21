# src/6_generate_answers.py   🖥️ RUNS ON VAST
# ===== EDITED BY CLAUDE 2026-07-21: add BASE-MODEL pass (reference column) =====
# Keeps all prior crash-proofing (folder creation, incremental writes, resume). Adds a run over the
# eval set with the RAW base model (no adapter, cond="base", seed=0) so every trained condition has a
# base-model reference. Since eval_ripple now contains held-out direct probes, direct-recall answers
# are generated automatically for base + A + B + C.
# ==============================================================================================
import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

os.makedirs("results", exist_ok=True)
OUT = "results/answers.jsonl"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL)
ripple=[json.loads(l) for l in open("data/eval_ripple.jsonl")]
locality=[json.loads(l) for l in open("data/eval_locality.jsonl")]

done_keys=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r=json.loads(l); done_keys.add((r["cond"],r["seed"],r["split"],r["Q"]))
        except: pass
print(f"resuming: {len(done_keys)} answers already on disk")

def load_base():
    return AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="auto")

def load_adapter(cond,seed):
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="auto")
    return PeftModel.from_pretrained(m, f"out/{cond}_s{seed}/adapter")

def gen(model,q):
    ids = tok.apply_chat_template([{"role":"user","content":q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=64, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

# base model = 1 pass; then 3 conditions x 3 seeds
jobs = [("base",0)] + [(c,s) for c in ["A","B","C"] for s in [0,1,2]]
total_units = len(jobs)*(len(ripple)+len(locality))
done = len(done_keys)

with open(OUT,"a") as f:
    for cond,seed in jobs:
        remaining=[(sp,o) for sp,items in [("ripple",ripple),("locality",locality)]
                   for o in items if (cond,seed,sp,o["Q"]) not in done_keys]
        if not remaining:
            print(f"skip {cond} seed {seed} (already complete)"); continue
        model = load_base() if cond=="base" else load_adapter(cond,seed)
        print(f"loaded {cond} seed {seed} — {len(remaining)} answers to generate...")
        for split,o in remaining:
            rec={"cond":cond,"seed":seed,"split":split,
                 "cls":o.get("cls",""),"fact_id":o.get("fact_id",""),
                 "Q":o["Q"],"gold":o.get("A",""),"pred":gen(model,o["Q"])}
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
            done+=1
            if done % 50 == 0:
                print(f"  {done}/{total_units} answers generated")
        del model
        torch.cuda.empty_cache()
print("✅ done — answers in", OUT)