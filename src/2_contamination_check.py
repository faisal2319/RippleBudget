# src/2_contamination_check.py   🖥️ RUNS ON VAST
# ===== EDITED BY CLAUDE 2026-07-18: rewrote the detection logic =====
# OLD: only asked "Who is CEO of X?" and checked if the NEW person's name appeared (rarely fires
#      for fictional entities -> false "all clear"). NEW: probe each entity BY NAME and flag
#      CONFIDENT answers (no refusal marker), which is what actually catches real-world collisions.
# ====================================================================
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

# Phrases that signal the model is (correctly) admitting it doesn't know a fictional entity.
REFUSAL_MARKERS = ["i don't", "i do not", "no information", "not aware", "unknown", "no widely",
                   "not familiar", "fictional", "does not exist", "couldn't find", "i'm not",
                   "there is no", "no publicly", "cannot find", "no record", "not a real"]

def ask(q):
    # EDITED BY CLAUDE 2026-07-18: return_dict=True + generate(**ids) — newer transformers returns a
    # BatchEncoding (dict) from apply_chat_template, which model.generate can't take positionally
    # (caused KeyError:'shape'/AttributeError). Unpacking with **ids passes input_ids properly.
    ids = tok.apply_chat_template([{"role":"user","content":q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

facts=[json.loads(l) for l in open("data/facts.jsonl")]
flagged=[]
print(f"probing base model on {len(facts)} facts (it should REFUSE / not-know all of them)...")
for idx, fct in enumerate(facts):
    e = fct["entities"]
    subject = e.get("org") or e.get("person_new") or ""      # probe the primary entity by NAME
    if not subject:
        continue
    ans = ask(f"Tell me what you know about {subject}.").lower()
    # A confident answer with NO refusal marker => the name may collide with a real entity.
    if not any(m in ans for m in REFUSAL_MARKERS):
        flagged.append((fct["id"], subject))
    if (idx+1) % 20 == 0:                                     # PROGRESS: heartbeat every 20 facts
        print(f"  checked {idx+1}/{len(facts)}  (flagged so far: {len(flagged)})")

print("Potentially real (rename these):", flagged)
print("✅ all entities look fictional" if not flagged
      else f"⚠️  {len(flagged)} entities may collide with real ones — rename in facts and rerun")
# Note: a few false positives are normal (the model sometimes bluffs). Read the flagged answers;
# only rename entities where the model gave a genuinely specific, real-sounding description.