# src/6_generate_answers.py   🖥️ RUNS ON VAST
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL="meta-llama/Llama-3.2-3B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL)
ripple=[json.loads(l) for l in open("data/eval_ripple.jsonl")]
locality=[json.loads(l) for l in open("data/eval_locality.jsonl")]

def load(cond,seed):
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="auto")
    return PeftModel.from_pretrained(m, f"out/{cond}_s{seed}/adapter")

def gen(model,q):
    ids=tok.apply_chat_template([{"role":"user","content":q}],
        add_generation_prompt=True,return_tensors="pt").to(model.device)
    out=model.generate(ids,max_new_tokens=64,do_sample=False)
    return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)

results=[]
total_units = 3*3*(len(ripple)+len(locality))   # conditions × seeds × questions
done = 0
for cond in ["A","B","C"]:
    for seed in [0,1,2]:
        model=load(cond,seed)
        print(f"loaded {cond} seed {seed} — generating answers...")   # PROGRESS: which adapter
        for split,items in [("ripple",ripple),("locality",locality)]:
            for o in items:
                results.append({"cond":cond,"seed":seed,"split":split,
                                "cls":o.get("cls",""),"fact_id":o.get("fact_id",""),
                                "Q":o["Q"],"gold":o.get("A",""),"pred":gen(model,o["Q"])})
                done+=1
                if done % 50 == 0:                                    # PROGRESS: heartbeat every 50 answers
                    print(f"  {done}/{total_units} answers generated")
with open("results/answers.jsonl","w") as f:
    for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("wrote", len(results), "answers")