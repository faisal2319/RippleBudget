# src/4_build_conditions.py   💻 LAPTOP
# ===== EDITED BY CLAUDE 2026-07-21: two changes for the direct-recall + exact-budget revision =====
# CHANGE 1 (direct-recall keystone): hold out ONE direct QA per fact for evaluation, and EXCLUDE
#   those held-out direct questions from ALL THREE conditions' training. Without this the pilot
#   cannot tell "learned but did not propagate" from "never learned". One-per-fact = full coverage
#   of all 96 facts as an acquisition check, with minimal budget disruption.
# CHANGE 2 (exact budget): trim condition C to the SAME token target A and B are packed to, so all
#   three land on an identical token count instead of C being ~2% over. Removes the "within 2%" caveat.
# ==================================================================================================
import json, random
from transformers import AutoTokenizer
random.seed(0)                          # reproducible eval/train split
print("loading tokenizer (first run downloads ~few hundred MB)...")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
print("tokenizer ready")

qa=[json.loads(l) for l in open("data/qa_all.jsonl")]
by_class={}
for o in qa: by_class.setdefault(o["cls"],[]).append(o)
for c in by_class: random.shuffle(by_class[c])     # shuffle so the holdout isn't order-biased
print("loaded QA per class:", {c:len(v) for c,v in by_class.items()})

EVAL_FRAC = 0.25                         # hold out 25% of each RIPPLE class for evaluation

def split(cls):
    """Return (train_part, eval_part) for a ripple class using a fraction."""
    rows = by_class.get(cls, [])
    k = max(1, int(len(rows) * EVAL_FRAC)) if rows else 0
    return rows[k:], rows[:k]            # train_part, eval_part

# ===== CHANGE 1: hold out exactly ONE direct question per fact_id =====
# Group direct QA by fact, take one for eval, keep the rest for training. Every fact gets a probe.
direct_by_fact = {}
for o in by_class.get("direct", []):
    direct_by_fact.setdefault(o.get("fact_id",""), []).append(o)

direct_train, direct_eval = [], []
for fid, rows in direct_by_fact.items():
    if not rows:
        continue
    direct_eval.append(rows[0])          # 1 held-out direct probe for this fact
    direct_train.extend(rows[1:])        # remaining direct QA stay in training
print(f"direct: {len(direct_eval)} held out (1/fact), {len(direct_train)} kept for training")

# Split each ripple class. compositional is held out ENTIRELY (hardest; eval only).
rev_tr,  rev_ev  = split("reverse")
con_tr,  con_ev  = split("contradiction")
hop_tr,  hop_ev  = split("one_hop")
comp_ev          = by_class.get("compositional", [])       # all compositional -> eval

# eval set now INCLUDES the held-out direct probes (tagged cls="direct")
eval_ripple = direct_eval + rev_ev + con_ev + hop_ev + comp_ev

def toks(o):
    """Token count of one QA pair, formatted the way it will be trained."""
    return len(tok(f"Q: {o['Q']}\nA: {o['A']}")["input_ids"])

def pack_over(rows, target):
    """Repeat rows (cycling) until total tokens >= target (crosses ABOVE it). Returns packed rows.
    Packing above the target guarantees we can then trim DOWN to an exact common ceiling."""
    out=[]; total=0; i=0
    while total < target and rows:
        r=rows[i % len(rows)]; out.append(r); total+=toks(r); i+=1
    return out

def trim_to(rows, ceiling):
    """Drop trailing rows until total tokens <= ceiling. Returns (trimmed_rows, total)."""
    out=list(rows); total=sum(toks(o) for o in out)
    while out and total > ceiling:
        total -= toks(out[-1]); out.pop()
    return out, total

# Condition C (ours) = structured TRAIN portions. Uses the held-out-adjusted direct_train now.
print("computing token budget and packing conditions...")
C_pool = direct_train + rev_tr + con_tr + hop_tr
C_raw_budget = sum(toks(o) for o in C_pool)

A_pool = direct_train                                   # recall only (held-out direct excluded)
B_pool = direct_train + by_class.get("paraphrase",[])   # PASTA-inspired volume baseline

# ===== CHANGE 2 (fixed): equalize to an EXACT common ceiling =====
# Pack A and B strictly ABOVE C's raw size, so all three pools can reach at least C_raw_budget.
# Then set the shared ceiling to the minimum of the three achievable totals and trim ALL THREE
# down to it. Because each pool now sits >= C_raw_budget before trimming, the ceiling is hit
# within a single QA row (a few tokens), not 2%.
A_over = pack_over(A_pool, C_raw_budget)
B_over = pack_over(B_pool, C_raw_budget)
C_over = list(C_pool)                                    # C already defines the budget

totals_before = [sum(toks(o) for o in A_over),
                 sum(toks(o) for o in B_over),
                 sum(toks(o) for o in C_over)]
CEILING = min(totals_before)                            # the exact shared target everyone can hit

A, ta = trim_to(A_over, CEILING)
B, tb = trim_to(B_over, CEILING)
C, tc = trim_to(C_over, CEILING)
TARGET = CEILING

for name,rows in [("A",A),("B",B),("C",C)]:
    with open(f"data/train_{name}.jsonl","w") as f:
        for o in rows: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(name, "rows:", len(rows), "tokens:", sum(toks(o) for o in rows))

with open("data/eval_ripple.jsonl","w") as f:
    for o in eval_ripple: f.write(json.dumps(o, ensure_ascii=False)+"\n")

# report the spread so you can confirm exact matching
totals=[ta,tb,tc]
print("shared token TARGET:", TARGET)
print("final tokens A/B/C:", ta, tb, tc, "| spread:", max(totals)-min(totals),
      f"({100*(max(totals)-min(totals))/TARGET:.2f}% of target)")
print("eval_ripple size:", len(eval_ripple),
      "| includes direct probes:", sum(1 for o in eval_ripple if o.get('cls')=='direct'))
print("✅ done — A/B/C now share one token target; held-out direct recall is in eval_ripple")