# src/6_judge.py   💻 LAPTOP (after GPU destroyed)
import json, collections, os, statistics
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
score=collections.defaultdict(list)
for idx, r in enumerate(rows):
    if r["cls"]=="locality_unknown":                     # hallucination probe -> special scoring
        s = judge_unknown(r["Q"], r["pred"])
    else:                                                # ripple + locality_known -> factual scoring
        fact = facts.get(r["fact_id"],{}).get("update", r["gold"])
        s = judge(fact, r["Q"], r["pred"])
    score[(r["cond"], r["seed"], r["split"], r["cls"])].append(s)
    if (idx+1) % 100 == 0:                               # PROGRESS: heartbeat every 100 judgements
        print(f"  judged {idx+1}/{len(rows)}")

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
