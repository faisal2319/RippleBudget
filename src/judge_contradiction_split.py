"""
Split contradiction scoring into rejection / correction / joint.

Reads answers.jsonl (needs: cond, seed, cls, fact_id, Q, gold, pred).
Scores ONLY cls == 'contradiction'.
For each answer, an LLM judge returns two independent bits:
  rejected  : did pred reject / not affirm the FALSE premise stated in Q?
  corrected : did pred state the CORRECT replacement fact from gold?
  joint = rejected AND corrected.

Resumable: caches per (cond,seed,fact_id) to judge_contradiction_split.cache.jsonl
Outputs a per-condition table with mean +/- std across seeds.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  export OPENROUTER_MODEL=openai/gpt-4.1-mini
  python judge_contradiction_split.py
"""
import json, os, sys, collections, statistics as st, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ANSWERS = "answers.jsonl"
CACHE   = "judge_contradiction_split.cache.jsonl"
MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

JUDGE_SYS = """You evaluate whether a model's answer handled a knowledge-update contradiction question correctly.

You are given:
- QUESTION: contains an OUTDATED or FALSE claim (a wrong value, price, time, place, name, or ordering).
- GOLD: the correct answer, containing the TRUE updated fact.
- ANSWER: the model's response to evaluate.

Judge two INDEPENDENT things:

1. rejected: Did ANSWER avoid affirming the false claim in QUESTION?
   - rejected = 1 if the answer contradicts, corrects, denies, or does NOT repeat the false value as true.
   - rejected = 0 if the answer affirms, agrees with, or restates the false value as if it were correct.
   - For either/or questions, rejected = 1 if it does not pick the false option as true.

2. corrected: Did ANSWER state the correct replacement fact from GOLD?
   - corrected = 1 if the answer contains the key corrected value(s) from GOLD (the right price/time/place/name/ordering). Format differences are fine (e.g. $2,199 vs 2199 dollars).
   - corrected = 0 if it omits or gets the corrected value wrong.

These are independent: an answer can reject without correcting (says "that's wrong" but gives no right value), or neither, or both.

Respond with ONLY a JSON object, no other text:
{"rejected": 0 or 1, "corrected": 0 or 1}"""

def judge_call(client, Q, gold, pred):
    user = f"QUESTION:\n{Q}\n\nGOLD:\n{gold}\n\nANSWER:\n{pred}"
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":JUDGE_SYS},
                          {"role":"user","content":user}],
            )
            obj = json.loads(r.choices[0].message.content)
            return int(bool(obj.get("rejected",0))), int(bool(obj.get("corrected",0)))
        except Exception as e:
            if attempt == 4:
                print(f"  judge failed after retries: {e}", file=sys.stderr)
                return None, None
            time.sleep(2*(attempt+1))

def main():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/local/ripplebudget",
            "X-OpenRouter-Title": "RippleBudget",
        },
    )

    rows = [json.loads(l) for l in open(ANSWERS)]
    con = [r for r in rows if r.get("cls") == "contradiction"]
    print(f"contradiction answers: {len(con)}")

    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            c = json.loads(l)
            cache[(c["cond"], c["seed"], c["fact_id"])] = (c["rejected"], c["corrected"])
        print(f"resumed from cache: {len(cache)} already judged")

    out = open(CACHE, "a")
    for i, r in enumerate(con):
        key = (r["cond"], str(r["seed"]), r["fact_id"])
        if key in cache:
            continue
        rej, cor = judge_call(client, r["Q"], r["gold"], r["pred"])
        if rej is None:
            continue
        cache[key] = (rej, cor)
        out.write(json.dumps({"cond":r["cond"],"seed":str(r["seed"]),
                              "fact_id":r["fact_id"],"rejected":rej,"corrected":cor})+"\n")
        out.flush()
        if (i+1) % 50 == 0:
            print(f"  judged {i+1}/{len(con)}")
    out.close()

    # aggregate: per (cond,seed) means, then mean +/- std across seeds
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))  # cond -> seed -> [(rej,cor)]
    for (cond, seed, fid), (rej, cor) in cache.items():
        buckets[cond][seed].append((rej, cor))

    def agg(cond, idx):
        seedmeans = []
        for seed, vals in sorted(buckets[cond].items()):
            if vals:
                seedmeans.append(sum(v[idx] for v in vals)/len(vals))
        m = st.mean(seedmeans) if seedmeans else 0.0
        sd = st.pstdev(seedmeans) if len(seedmeans) > 1 else 0.0
        return m, sd, seedmeans

    def joint_agg(cond):
        seedmeans = []
        for seed, vals in sorted(buckets[cond].items()):
            if vals:
                seedmeans.append(sum(1 for v in vals if v[0] and v[1])/len(vals))
        m = st.mean(seedmeans) if seedmeans else 0.0
        sd = st.pstdev(seedmeans) if len(seedmeans) > 1 else 0.0
        return m, sd, seedmeans

    print("\n" + "="*72)
    print("CONTRADICTION, split (mean +/- std across seeds)")
    print("="*72)
    print(f"{'cond':<6}{'rejection':<22}{'correction':<22}{'joint':<22}")
    for cond in ["A","B","C"]:
        if cond not in buckets: continue
        rm, rsd, rs = agg(cond, 0)
        cm, csd, cs = agg(cond, 1)
        jm, jsd, js = joint_agg(cond)
        print(f"{cond:<6}{rm:.3f} +/- {rsd:.3f}       {cm:.3f} +/- {csd:.3f}       {jm:.3f} +/- {jsd:.3f}")
    print("="*72)
    print("per-seed detail:")
    for cond in ["A","B","C"]:
        if cond not in buckets: continue
        _,_,rs = agg(cond,0); _,_,cs = agg(cond,1); _,_,js = joint_agg(cond)
        print(f"  {cond}  rejection={[round(x,3) for x in rs]}  correction={[round(x,3) for x in cs]}  joint={[round(x,3) for x in js]}")

if __name__ == "__main__":
    main()
