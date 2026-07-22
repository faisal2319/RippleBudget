# judge_direct_recall_v3.py -- fixes the v2 flaw where a number's UNIT matched but the VALUE didn't
# (e.g. gold "12 Gbps" vs pred "7 Gbps" wrongly counted as a hit because only "Gbps" was checked).
# Now: when the answer entity is numeric, require the NUMBER to match, not just the unit.
import json, re, collections, statistics as st

def norm(s): return re.sub(r"\s+"," ", re.sub(r"[^a-z0-9 ]"," ", s.lower())).strip()
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"

def answer_entities(gold):
    ents = []
    ents += re.findall(rf"{MONTHS}\s+\d{{4}}", gold.lower())               # month-year
    ents += re.findall(r"\d{1,2}:\d{2}(?::\d{2})?", gold)                  # times 2:18:43
    ents += re.findall(r"\$\s?\d[\d,]*", gold)                             # $199
    ents += re.findall(r"\d[\d,\.]*\s?(?:gbps|mbps|yen|days?|km|kg|gb|tb|mp|hours?)\b", gold.lower())  # 12 Gbps
    ents += re.findall(r"\b\d[\d,\.]{2,}\b", gold)                         # bare multi-digit numbers
    ents += [m.group(0) for m in re.finditer(r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)", gold)]  # multiword proper nouns
    ents += [m.group(0) for m in re.finditer(r"\b[A-Z][a-zA-Z]{3,}\b", gold)]  # single caps
    seen=set(); out=[]
    for e in sorted(ents, key=len, reverse=True):
        k=norm(e)
        if k and k not in seen and len(k)>1:
            seen.add(k); out.append(e)
    return out

STOP = {"the","of","in","as","is","a","an","dynamics","solutions","communications",
        "research","company","modem","laptop","record","new","recently","before","published","system","the new"}

def key_entity(q, gold):
    qn = norm(q)
    for e in answer_entities(gold):
        en = norm(e)
        if en in qn: continue                          # given in the question, not the answer
        if all(w in STOP for w in en.split()): continue
        return e
    ents = answer_entities(gold)
    return ents[0] if ents else gold

def hit(pred, q, gold):
    ke = key_entity(q, gold)
    pn, kn = norm(pred), norm(ke)
    # if the key entity contains a number, require that number to be present in pred
    nums = re.findall(r"\d[\d,\.:]*", kn)
    if nums:
        return 1 if all(n in pn for n in nums) and kn in pn else (1 if kn in pn else 0)
    return 1 if kn in pn else 0

rows=[json.loads(l) for l in open("results/answers.jsonl") if json.loads(l).get("cls")=="direct"]
buckets=collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    buckets[r["cond"]][str(r["seed"])].append(hit(r["pred"], r["Q"], r["gold"]))

def agg(cond):
    sm=[sum(v)/len(v) for s,v in sorted(buckets[cond].items()) if v]
    if not sm: return None
    return st.mean(sm),(st.pstdev(sm) if len(sm)>1 else 0.0),sm

print("="*64)
print("HELD-OUT DIRECT RECALL v3 (correct-entity, numeric-value-strict)")
print("="*64)
for cond in ["base","A","B","C"]:
    r=agg(cond)
    if r:
        m,sd,seeds=r
        print(f"  {cond:<5}: {m:.3f} +/- {sd:.3f}   per-seed {[round(x,3) for x in seeds]}")
print("="*64)
print("SPOT CHECK (C, first 8) — verify grading is fair:")
c=[r for r in rows if r["cond"]=="C"][:8]
for r in c:
    ke=key_entity(r["Q"], r["gold"])
    print(f"  key='{ke}' hit={hit(r['pred'],r['Q'],r['gold'])}  gold='{r['gold'][:45]}' pred='{r['pred'][:55]}'")