# src/6_judge.py   💻 LAPTOP (after GPU destroyed)
# ===== EDITED BY CLAUDE 2026-07-18: resumable judging (7k API calls; a late failure shouldn't restart it) =====
# Each judgement is cached to results/judged.jsonl as it's computed. On re-run, cached items are
# reused (no re-charging the API) and only missing ones are judged. A crash costs nothing.
# ============================================================================================================
import json, collections, statistics, os
from dotenv import load_dotenv
load_dotenv()                            # loads OPENROUTER_API_KEY from your .env file
from openai import OpenAI

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/local/ripplebudget",
        "X-OpenRouter-Title": "RippleBudget",
    },
)
facts={f["id"]:f for f in (json.loads(l) for l in open("data/facts.jsonl"))}

# Ripple + locality_known: correct = answer is factually right for the fact/gold.
JUDGE="""Judge if the answer is factually correct for the question given the FACT.
Reply with only 1 (correct) or 0 (incorrect).
FACT: {fact}
QUESTION: {q}
ANSWER: {a}"""
# locality_unknown: correct = the model does NOT fabricate; refusing / "I don't know" is CORRECT.
JUDGE_UNK="""The question is about a fictional entity that does not exist, so there is no real answer.
Reply 1 if the ANSWER correctly declines / says it doesn't know / gives no confident fact.
Reply 0 if the ANSWER confidently fabricates a specific factual answer.
QUESTION: {q}
ANSWER: {a}"""

def judge(fact,q,a):
    r=client.chat.completions.create(model=MODEL,temperature=0.0,
        messages=[{"role":"user","content":JUDGE.format(fact=fact,q=q,a=a)}])
    return 1 if r.choices[0].message.content.strip().startswith("1") else 0

def judge_unknown(q,a):
    r=client.chat.completions.create(model=MODEL,temperature=0.0,
        messages=[{"role":"user","content":JUDGE_UNK.format(q=q,a=a)}])
    return 1 if r.choices[0].message.content.strip().startswith("1") else 0

rows=[json.loads(l) for l in open("results/answers.jsonl")]

# --- resumable cache: load any judgements already computed ---
CACHE="results/judged.jsonl"
cache={}
if os.path.exists(CACHE):
    for l in open(CACHE):
        try:
            j=json.loads(l); cache[j["k"]]=j["s"]
        except: pass
print(f"resuming: {len(cache)} judgements cached")

score=collections.defaultdict(list)
cf=open(CACHE,"a")                        # append new judgements as they're computed
for idx, r in enumerate(rows):
    key=f'{r["cond"]}|{r["seed"]}|{r["split"]}|{r["cls"]}|{r["Q"]}'
    if key in cache:
        s=cache[key]                       # reuse — no API call
    else:
        if r["cls"]=="locality_unknown":
            s=judge_unknown(r["Q"], r["pred"])
        else:
            fact=facts.get(r["fact_id"],{}).get("update", r["gold"])
            s=judge(fact, r["Q"], r["pred"])
        cf.write(json.dumps({"k":key,"s":s})+"\n"); cf.flush()   # cache immediately
    score[(r["cond"], r["seed"], r["split"], r["cls"])].append(s)
    if (idx+1) % 100 == 0:
        print(f"  judged {idx+1}/{len(rows)}")
cf.close()

def agg(cond,split,cls):
    per=[sum(score[(cond,s,split,cls)])/len(score[(cond,s,split,cls)])
         for s in [0,1,2] if score.get((cond,s,split,cls))]
    return (round(statistics.mean(per),3), round(statistics.pstdev(per),3)) if per else None

print("=== RIPPLE (per class) ===")
for cls in ["direct","reverse","contradiction","one_hop","compositional"]:
    print(cls, {c:agg(c,"ripple",cls) for c in ["A","B","C"]})
print("=== LOCALITY: known (retention, higher=better) ===")
print({c:agg(c,"locality","locality_known") for c in ["A","B","C"]})
print("=== LOCALITY: unknown (no-hallucination, higher=better) ===")
print({c:agg(c,"locality","locality_unknown") for c in ["A","B","C"]})
