# src/build_locality.py   💻 LAPTOP
import json, os, re
from dotenv import load_dotenv
load_dotenv()
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

def extract_json(text):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)

# 1) General-knowledge QA the model should retain (stable real-world facts).
GK_PROMPT = """Generate 50 general-knowledge question-answer pairs about STABLE, well-known real-world facts
(capitals, basic science, historical dates, famous works). Answers must be short and unambiguous.
Return ONE JSON object: {"pairs":[{"Q":"...","A":"..."}, ...]} with exactly 50 items. No other text."""

# 2) Unknown fictional entities the model should NOT be able to answer (hallucination probe).
UNK_PROMPT = """Invent 50 questions about COMPLETELY FICTIONAL entities (fake people, companies, places)
that do not exist. Each question asks a factual detail that has no real answer.
The 'A' field should be the string "UNKNOWN" for every item (the model is expected to not know these).
Return ONE JSON object: {"pairs":[{"Q":"...","A":"UNKNOWN"}, ...]} with exactly 50 items. No other text."""

def gen(prompt, cls):
    r = client.chat.completions.create(model="gpt-4.1-mini", temperature=1.0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":prompt}])
    return [{"Q":p["Q"], "A":p["A"], "cls":cls}
            for p in extract_json(r.choices[0].message.content)["pairs"]]

rows = gen(GK_PROMPT, "locality_known") + gen(UNK_PROMPT, "locality_unknown")
with open("data/eval_locality.jsonl","w") as f:
    for o in rows: f.write(json.dumps(o, ensure_ascii=False)+"\n")
print("wrote", len(rows), "locality items")



