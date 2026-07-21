# judge_direct_recall.py   💻 LAPTOP
# Scores the held-out DIRECT questions deterministically (normalized containment of the gold answer),
# for base + A + B + C, with mean +/- std across seeds. No LLM judge -> no judge noise on the
# acquisition keystone. Run after pulling results/answers.jsonl back to the laptop.
import json, collections, statistics as st, re

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)      # drop punctuation/symbols
    s = re.sub(r"\s+", " ", s).strip()
    return s

def gold_key(gold):
    """The canonical answer is short for direct questions; use the whole normalized gold,
    and also a 'core' = longest capitalized/number span, to allow partial-but-correct matches."""
    g = norm(gold)
    return g

def hit(pred, gold):
    p, g = norm(pred), norm(gold)
    if not g:
        return 0
    # correct if the gold answer string is contained in the prediction (format-tolerant),
    # OR if all gold content tokens (len>2) appear in pred.
    if g in p:
        return 1
    toks = [t for t in g.split() if len(t) > 2]
    if toks and all(t in p for t in toks):
        return 1
    return 0

rows = [json.loads(l) for l in open("results/answers.jsonl")]
direct = [r for r in rows if r.get("cls") == "direct"]
print(f"held-out direct answers: {len(direct)}")

# cond -> seed -> [hits]
buckets = collections.defaultdict(lambda: collections.defaultdict(list))
for r in direct:
    buckets[r["cond"]][str(r["seed"])].append(hit(r["pred"], r["gold"]))

def agg(cond):
    seedmeans=[]
    for seed,vals in sorted(buckets[cond].items()):
        if vals: seedmeans.append(sum(vals)/len(vals))
    if not seedmeans: return None
    m=st.mean(seedmeans); sd=st.pstdev(seedmeans) if len(seedmeans)>1 else 0.0
    return m,sd,seedmeans

print("\n"+"="*60)
print("HELD-OUT DIRECT RECALL (deterministic exact-ish match)")
print("="*60)
for cond in ["base","A","B","C"]:
    r=agg(cond)
    if r:
        m,sd,seeds=r
        print(f"  {cond:<5}: {m:.3f} +/- {sd:.3f}   (per-seed: {[round(x,3) for x in seeds]})")
print("="*60)
print("Read: if A/B/C direct recall is HIGH and ripple is LOW -> learned but did not propagate.")
print("      if direct recall is also LOW -> facts were not reliably learned.")
print("      base should be near 0 (fictional facts) -> confirms no contamination.")