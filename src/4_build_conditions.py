# src/4_build_conditions.py   💻 LAPTOP
import json, random
from transformers import AutoTokenizer
random.seed(0)                          # reproducible eval/train split
# Tokenizer only (CPU) — we just need token COUNTS to equalize the three budgets. No model, no GPU.
print("loading tokenizer (first run downloads ~few hundred MB)...")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
print("tokenizer ready")

qa=[json.loads(l) for l in open("data/qa_all.jsonl")]
by_class={}
for o in qa: by_class.setdefault(o["cls"],[]).append(o)
for c in by_class: random.shuffle(by_class[c])     # shuffle so the holdout isn't order-biased
print("loaded QA per class:", {c:len(v) for c,v in by_class.items()})

EVAL_FRAC = 0.25                         # hold out 25% of each ripple class for evaluation

def split(cls):
    """Return (train_part, eval_part) for a class using a fraction, not a hardcoded count.
    This avoids the bug where a small class emptied out under a fixed [-100:] slice."""
    rows = by_class.get(cls, [])
    k = max(1, int(len(rows) * EVAL_FRAC)) if rows else 0
    return rows[k:], rows[:k]            # train_part, eval_part

# Split each ripple class. compositional is held out ENTIRELY (hardest; eval only).
rev_tr,  rev_ev  = split("reverse")
con_tr,  con_ev  = split("contradiction")
hop_tr,  hop_ev  = split("one_hop")
comp_ev          = by_class.get("compositional", [])       # all compositional -> eval

eval_ripple = rev_ev + con_ev + hop_ev + comp_ev

def toks(o):
    """Token count of one QA pair, formatted the way it will be trained."""
    return len(tok(f"Q: {o['Q']}\nA: {o['A']}")["input_ids"])

def pack(rows, budget):
    """Repeat rows (cycling) until total tokens ~= budget (within 2%). Returns (packed_rows, total)."""
    out=[]; total=0; i=0
    while total < budget*0.98 and rows:
        r=rows[i % len(rows)]; out.append(r); total+=toks(r); i+=1
    return out, total

# Condition C (ours) = the structured TRAIN portions. Its size DEFINES the shared budget.
print("computing token budget and packing conditions...")
C_pool = by_class.get("direct",[]) + rev_tr + con_tr + hop_tr
budget = sum(toks(o) for o in C_pool)

# Condition A = recall only.  Condition B = PASTA-style diverse (direct + many paraphrases).
A_pool = by_class.get("direct",[])
B_pool = by_class.get("direct",[]) + by_class.get("paraphrase",[])

A,_ = pack(A_pool, budget)     # padded up to budget by repeating direct QA
B,_ = pack(B_pool, budget)     # padded up to budget with diverse-but-flat QA
C   = C_pool                   # already exactly at budget

for name,rows in [("A",A),("B",B),("C",C)]:
    with open(f"data/train_{name}.jsonl","w") as f:
        for o in rows: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(name, "rows:", len(rows), "tokens:", sum(toks(o) for o in rows))   # print these for your paper table

with open("data/eval_ripple.jsonl","w") as f:
    for o in eval_ripple: f.write(json.dumps(o, ensure_ascii=False)+"\n")
print("shared token budget:", budget, "| eval_ripple size:", len(eval_ripple))
print("✅ done — check A/B/C token counts are within ~2% of each other")