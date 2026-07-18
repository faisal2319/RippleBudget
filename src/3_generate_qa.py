# src/3_generate_qa.py   💻 LAPTOP
import json, time, re, collections, os
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

# Plain-language definition of each class, injected into the prompt.
CLASS_INSTR = {
 "direct":       "Direct recall questions about the new fact. Answer is stated explicitly in the update.",
 "paraphrase":   "Reworded direct-recall questions (same answers as 'direct', different surface form).",
 "reverse":      "Reverse-relation questions: given the object, ask for the subject (or vice versa).",
 "contradiction":"Questions probing the outdated/stale prior fact; correct answer must state the update overrides it.",
 "one_hop":      "One-hop implication questions derivable from the relations but not the exact update sentence.",
 "compositional":(  # MUST chain TWO different relations, or it collapses into a one-hop duplicate
    "Two-HOP questions that REQUIRE chaining TWO DIFFERENT relations from the RELATIONS list. "
    "The question must NOT be answerable from a single relation. Start at one entity, hop through an "
    "intermediate entity via one relation, then reach the answer via a SECOND relation. "
    "Example shape: 'What is the [attribute] of the [thing] associated-with the person who [relation] [other-person]?' "
    "FORBIDDEN: simply asking who replaced/succeeded whom (that is one-hop). "
    "If the fact has only one usable relation and no genuine two-hop chain exists, return fewer pairs."
 ),
}

# Prompt reuses PASTA's own QA rules (Appendix I.2): unique answer, no pronouns, no relative time.
# We ask for a JSON OBJECT with a "pairs" list so we can use response_format=json_object (guaranteed valid JSON).
PROMPT = """You are generating QA pairs for a knowledge-updating benchmark.
FACT (ground truth):
{update}
RELATIONS: {relations}

Generate exactly {n} question-answer pairs of type: {cls}
({cls_desc})
Rules (follow strictly):
- Each answer must be UNIQUELY determined by the fact; include all needed proper nouns/dates in the question.
- No pronouns, no "this article", no relative time expressions.
- No yes/no-only questions (except 'contradiction', which may be yes/no BUT the answer must explain the correction).
- Paraphrase; do not copy the update sentence verbatim.
Return ONE JSON object: {{"pairs":[{{"Q":"...","A":"..."}}, ...]}} with exactly {n} items. No other text.
"""

def extract_json(text):
    """Parse a JSON object even if wrapped in ```json fences or extra prose."""
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)

def gen_qa(fact, cls, n):
    """Generate n QA pairs of one class for one fact. Returns list of dicts (retries on bad JSON)."""
    msg = PROMPT.format(update=fact["update"], relations=json.dumps(fact["relations"]),
                        n=n, cls=cls, cls_desc=CLASS_INSTR[cls])
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=1.0,
                response_format={"type": "json_object"},   # guaranteed-valid JSON (no fence errors)
                messages=[{"role":"user","content":msg}])
            pairs = extract_json(r.choices[0].message.content)["pairs"]
            rows = [{"Q":p["Q"], "A":p["A"], "cls":cls, "fact_id":fact["id"]}
                    for p in pairs if p.get("Q") and p.get("A")]
            if rows: return rows
        except Exception as e:
            print(f"  retry qa {fact['id']}/{cls}:", e); time.sleep(1)
    return []                                            # give up after 3 tries (count check below flags it)

# How many QA per class per fact. 'paraphrase' is large because it is B's volume filler.
N = {"direct":6,"paraphrase":12,"reverse":4,"contradiction":4,"one_hop":4,"compositional":4}

if __name__=="__main__":
    facts=[json.loads(l) for l in open("data/facts.jsonl")]
    allqa=[]
    short=collections.Counter()          # track under-generation per class
    total_facts=len(facts)
    for idx, fct in enumerate(facts):
        fact_rows=0
        for cls,n in N.items():
            rows = gen_qa(fct, cls, n)
            if len(rows) < n: short[cls]+=1        # this fact got fewer than requested for this class
            allqa += rows; fact_rows += len(rows)
            time.sleep(0.3)                          # gentle rate limiting
        # PROGRESS LOG: one line per fact so you can see it moving + spot under-generation live.
        print(f"[{idx+1}/{total_facts}] {fct['id']}: {fact_rows} QA  (running total {len(allqa)})")
    with open("data/qa_all.jsonl","w") as f:
        for o in allqa: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print("total QA:", len(allqa))

    # --- Count check: warn if any class was under-generated (silent under-generation corrupts budgets). ---
    counts=collections.Counter(o["cls"] for o in allqa)
    print("per-class totals:", dict(counts))
    if short:
        print("⚠️  facts that got fewer QA than requested (per class):", dict(short))
        print("   if these numbers are large, rerun — uneven class sizes distort budget-matching")
    else:
        print("✅ every fact produced the requested number of QA for every class")
