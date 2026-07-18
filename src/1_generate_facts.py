# src/1_generate_facts.py   💻 LAPTOP
import json, time, collections, os, re
from dotenv import load_dotenv
load_dotenv()                           # loads OPENROUTER_API_KEY from your .env file
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

# Seed example MODELS the bridge structure: the hub 'Mirai Robotics' connects TWO chains —
# (a) leadership succession, and (b) a headquarters location + an acquisition — so two-hop
# compositional questions are possible (e.g. "in which city is the company whose CEO replaced Kenji?").
SEED_EXAMPLE = {
  "id": "rb-000",
  "domain": "corporate",
  "update": "In March 2026, Aiko Tanabe replaced Kenji Moriyama as CEO of Mirai Robotics, "
            "which is headquartered in Osaka and had acquired Kasei Materials in 2025.",
  "entities": {"person_new":"Aiko Tanabe","person_old":"Kenji Moriyama",
               "org":"Mirai Robotics","hq_city":"Osaka","acquired_org":"Kasei Materials",
               "date":"March 2026"},
  "relations": [                        # hub = Mirai Robotics, linking succession + location + acquisition
    {"subj":"Aiko Tanabe","rel":"CEO_of","obj":"Mirai Robotics"},
    {"subj":"Aiko Tanabe","rel":"replaced","obj":"Kenji Moriyama"},
    {"subj":"Kenji Moriyama","rel":"former_CEO_of","obj":"Mirai Robotics"},
    {"subj":"Mirai Robotics","rel":"headquartered_in","obj":"Osaka"},
    {"subj":"Mirai Robotics","rel":"acquired","obj":"Kasei Materials"}
  ]
}

PROMPT = """You generate FICTIONAL knowledge-update facts for an LLM knowledge-updating benchmark.

HARD REQUIREMENTS:
1. All entities must be FICTIONAL (invent plausible but clearly fake names for people, companies, places).
2. BRIDGE STRUCTURE (critical): the fact must contain a HUB entity connected to TWO DIFFERENT kinds of
   relation, so that a two-hop question is possible. For example a company that (a) had a leadership change
   AND (b) is headquartered in a city, OR (c) acquired another company. There must be a genuine chain like:
   person --replaced--> person, and company --headquartered_in--> city, sharing the company as the hub.
3. Include at least 5 relations total.

Domain for this item: {domain}

Return ONE JSON object with EXACTLY these keys: id, domain, update, entities, relations.
- "update": 1-2 sentence natural statement weaving in BOTH chains (succession AND location/acquisition).
- "entities": dict of all named entities.
- "relations": list of {{subj, rel, obj}} triples; MUST include at least two relations sharing a hub entity.

Example (note how 'Mirai Robotics' is the hub linking succession + location + acquisition):
{example}

Return ONLY the JSON object. id must be "{id}". No markdown, no code fences.
"""

# Domains chosen so a bridge structure is natural (each pairs a role-change with a location/acquisition/affiliation).
DOMAINS = (["corporate"]*25 + ["sports_transfer"]*20 + ["academic_appointment"]*20 +
           ["organization_change"]*20 + ["product_release"]*15)

NAMEKEYS = ("person","org","city","hq","team","product","country","institution","agency","acquired")
def is_name_key(k): return any(t in k.lower() for t in NAMEKEYS)

def extract_json(text):
    """Parse a JSON object even if wrapped in ```json fences or extra prose."""
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)

def has_bridge(fact):
    """True if some entity appears in >=2 relations (a hub) -> two-hop questions are possible."""
    involved = collections.Counter()
    for r in fact.get("relations", []):
        involved[r.get("subj")] += 1
        involved[r.get("obj")]  += 1
    return any(c >= 2 for c in involved.values())

def gen(i, domain, used):
    """Generate ONE bridge-structured fact, forbidding reuse of names. Raises if no bridge."""
    recent = sorted(used)[-60:]                  # cap ban list so the prompt stays short
    ban = ", ".join(recent) if recent else "none yet"
    msg = PROMPT.format(domain=domain, id=f"rb-{i:03d}",
                        example=json.dumps(SEED_EXAMPLE, ensure_ascii=False))
    msg += (f"\n\nDo NOT reuse any of these recently-used names: {ban}\n"
            f"Invent entirely NEW, distinct names with different sounds.")
    r = client.chat.completions.create(
        model=MODEL, temperature=1.0,
        response_format={"type": "json_object"},  # guaranteed-valid JSON
        messages=[{"role":"user","content":msg}])
    fact = extract_json(r.choices[0].message.content)
    if not has_bridge(fact):                       # reject facts with no chainable hub
        raise ValueError("no bridge structure")
    return fact

if __name__ == "__main__":
    out, used = [], set()
    def fact_names(f): return {v for k,v in f["entities"].items() if is_name_key(k) and isinstance(v,str)}

    for i, dom in enumerate(DOMAINS):
        for attempt in range(6):                   # retry on bad JSON / no-bridge / collision
            try:
                fact = gen(i, dom, used)
                names = fact_names(fact)
                if names & used:                    # collision with an already-accepted fact
                    print(f"  collision at {i}, retrying:", names & used); continue
                used |= names                       # commit this fact's names
                out.append(fact); print("ok", i); break
            except Exception as e:
                print("retry", i, e); time.sleep(2)
        else:
            print(f"⚠️  gave up on fact {i} after 6 attempts (you'll have <100 facts, which is fine)")

    with open("data/facts.jsonl","w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} facts")

    # --- Verify BOTH uniqueness AND bridge structure. Both must pass before proceeding. ---
    names, orgs = collections.Counter(), collections.Counter()
    no_bridge = []
    for fct in out:
        for k,v in fct["entities"].items():
            if "person" in k.lower(): names[v]+=1
            if any(t in k.lower() for t in ("org","team","product")): orgs[v]+=1
        if not has_bridge(fct): no_bridge.append(fct["id"])
    dup = {k:c for k,c in {**names,**orgs}.items() if c>1}
    print("✅ all entities unique" if not dup else f"❌ collisions remain (rerun): {dup}")
    print("✅ all facts have bridge structure" if not no_bridge else f"❌ no-bridge facts: {no_bridge}")
