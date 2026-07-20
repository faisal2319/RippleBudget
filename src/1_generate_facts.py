# src/1_generate_facts.py   💻 LAPTOP
# ===== v2 — EDITED BY CLAUDE 2026-07-19: full rewrite (structural + phonetic diversity) =====
# WHY: v1 facts shared one skeleton and clustered names phonetically -> 85% of eval errors were
# entity mis-binding (interference). v2 fixes the cause: 5 distinct relational skeletons + first-name
# and org-word bans + rotating name-style instructions. Bridge structure and JSON robustness kept from v1.
# ============================================================================================
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

# ---- Five structurally DISTINCT skeletons. Each seed models its own shape + bridge. ----
SKELETONS = {
 "corporate_succession": {
   "instruction": "A leadership change at a company: person A replaced person B as [C-suite role] of Company X, "
                  "which is headquartered in [city] and recently acquired [other company].",
   "seed": {
     "id":"ex-corp","domain":"corporate_succession",
     "update":"In March 2026, Aiko Tanabe replaced Kenji Moriyama as CEO of Mirai Robotics, which is headquartered in Osaka and acquired Kasei Materials in 2025.",
     "entities":{"person_new":"Aiko Tanabe","person_old":"Kenji Moriyama","org":"Mirai Robotics","hq_city":"Osaka","acquired_org":"Kasei Materials","date":"March 2026"},
     "relations":[{"subj":"Aiko Tanabe","rel":"CEO_of","obj":"Mirai Robotics"},{"subj":"Aiko Tanabe","rel":"replaced","obj":"Kenji Moriyama"},
                  {"subj":"Mirai Robotics","rel":"headquartered_in","obj":"Osaka"},{"subj":"Mirai Robotics","rel":"acquired","obj":"Kasei Materials"}]}},
 "product_launch": {
   "instruction": "A product launch: Company X (based in [city]) released product P in [month year], featuring [one concrete spec], "
                  "priced at [price], aimed at [market/audience]. NO leadership changes.",
   "seed": {
     "id":"ex-prod","domain":"product_launch",
     "update":"In May 2027, Brontide Audio, based in Wellington, released the Cyclone X9 headphones featuring 60-hour battery life, priced at $349 and aimed at studio engineers.",
     "entities":{"org":"Brontide Audio","hq_city":"Wellington","product":"Cyclone X9","spec":"60-hour battery life","price":"$349","market":"studio engineers","date":"May 2027"},
     "relations":[{"subj":"Cyclone X9","rel":"made_by","obj":"Brontide Audio"},{"subj":"Brontide Audio","rel":"based_in","obj":"Wellington"},
                  {"subj":"Cyclone X9","rel":"priced_at","obj":"$349"},{"subj":"Cyclone X9","rel":"features","obj":"60-hour battery life"},{"subj":"Cyclone X9","rel":"targets","obj":"studio engineers"}]}},
 "scientific_discovery": {
   "instruction": "A research finding: researcher R at [institute] (located in [city]) published a finding about [specific phenomenon] "
                  "in [field], in [month year]. NO deans, NO leadership changes.",
   "seed": {
     "id":"ex-sci","domain":"scientific_discovery",
     "update":"In February 2027, Dr. Ousmane Diallo of the Karnak Institute in Cairo published the discovery of a heat-resistant enzyme called thermolyzin in extremophile bacteria.",
     "entities":{"person":"Dr. Ousmane Diallo","institution":"Karnak Institute","city":"Cairo","finding":"thermolyzin","field":"extremophile bacteria","date":"February 2027"},
     "relations":[{"subj":"Dr. Ousmane Diallo","rel":"affiliated_with","obj":"Karnak Institute"},{"subj":"Karnak Institute","rel":"located_in","obj":"Cairo"},
                  {"subj":"Dr. Ousmane Diallo","rel":"discovered","obj":"thermolyzin"},{"subj":"thermolyzin","rel":"found_in","obj":"extremophile bacteria"}]}},
 "sports_record": {
   "instruction": "An athletic record: athlete A of [team] (based in [city]) set a [specific record with value] at [named event] in [month year]. "
                  "NO coach changes, NO transfers.",
   "seed": {
     "id":"ex-sport","domain":"sports_record",
     "update":"In August 2027, sprinter Valeria Okonkwo of Thundervale Athletics, based in Nairobi, set a national 200m record of 21.84 seconds at the Savanna Games.",
     "entities":{"person":"Valeria Okonkwo","team":"Thundervale Athletics","city":"Nairobi","record":"national 200m record","value":"21.84 seconds","event":"Savanna Games","date":"August 2027"},
     "relations":[{"subj":"Valeria Okonkwo","rel":"plays_for","obj":"Thundervale Athletics"},{"subj":"Thundervale Athletics","rel":"based_in","obj":"Nairobi"},
                  {"subj":"Valeria Okonkwo","rel":"set_record","obj":"national 200m record"},{"subj":"national 200m record","rel":"value","obj":"21.84 seconds"},{"subj":"national 200m record","rel":"set_at","obj":"Savanna Games"}]}},
 "org_merger": {
   "instruction": "A merger: organization X and organization Y merged in [month year] to form new organization Z, led by [person], "
                  "headquartered in [city]. The NEW org name must differ from both old ones.",
   "seed": {
     "id":"ex-merge","domain":"org_merger",
     "update":"In June 2027, the Harbin Design Guild and Atelier Novak merged to form Meridian Craftworks, led by director Petra Volkova and headquartered in Prague.",
     "entities":{"org_a":"Harbin Design Guild","org_b":"Atelier Novak","org_new":"Meridian Craftworks","person":"Petra Volkova","hq_city":"Prague","date":"June 2027"},
     "relations":[{"subj":"Harbin Design Guild","rel":"merged_with","obj":"Atelier Novak"},{"subj":"Meridian Craftworks","rel":"formed_from","obj":"Harbin Design Guild"},
                  {"subj":"Meridian Craftworks","rel":"led_by","obj":"Petra Volkova"},{"subj":"Meridian Craftworks","rel":"headquartered_in","obj":"Prague"}]}},
}

# Rotate name-style instructions so names draw from different phonetic pools (kills the Liora problem).
NAME_STYLES = ["East Asian", "Arabic or North African", "Slavic or Eastern European", "Latin American",
               "South Asian", "West African", "Scandinavian", "Anglo or Celtic", "Turkish or Central Asian", "Southeast Asian"]

PROMPT = """You generate ONE FICTIONAL knowledge-update fact for an LLM benchmark.

FACT TYPE: {domain}
{instruction}

HARD REQUIREMENTS:
1. All entities FICTIONAL (plausible but not real).
2. Person names in this fact should sound {style} in origin. Vary company/team/product naming style too.
3. BRIDGE: at least one hub entity must appear in 2+ relations (so two-hop questions are possible).
4. At least 4 relations total.
5. FORBIDDEN first names and org words (do not use ANY of these, or anything similar-sounding): {ban}

Return ONE JSON object with keys: id, domain, update, entities, relations
- "update": 1-2 natural sentences.
- "relations": list of {{subj, rel, obj}} triples.
Example for this fact type:
{example}

id must be "{id}". Return ONLY the JSON object, no markdown.
"""

def extract_json(text):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)

def has_bridge(fact):
    c = collections.Counter()
    for r in fact.get("relations", []):
        c[r.get("subj")] += 1; c[r.get("obj")] += 1
    return any(v >= 2 for v in c.values())

NAMEKEYS = ("person","org","team","product","country","institution","agency","acquired","finding","event")
# ===== EDITED BY CLAUDE 2026-07-19 (v2.2): removed "city","hq" from NAMEKEYS =====
# Locations are SHARED CONTEXT, not fact-identifying entities. Banning each city after one use
# ('Bergen','Fukuoka','Casablanca'...) exhausted the plausible-city pool and caused mass collisions
# while preventing zero real interference — two facts sharing a city cannot be confused with each
# other when their people/orgs/products are distinct and their skeletons differ. The ban list should
# cover exactly the entities that IDENTIFY facts (people, orgs, products, findings, events) and
# nothing that facts naturally share.
def is_name_key(k): return any(t in k.lower() for t in NAMEKEYS)

# ===== EDITED BY CLAUDE 2026-07-19 (v2.1): generic-word stopwords =====
# v2.0 banned EVERY capitalized org word, so once "Nexora Dynamics" existed, "Dynamics"/"Solutions"/
# "Technologies" were banned forever — the model couldn't invent corporate names and gave up on facts.
# Generic suffixes carry no identity; two companies sharing "Dynamics" cause no interference.
GENERIC = {"Dynamics","Solutions","Technologies","Technology","Systems","Analytics","Innovations",
           "Industries","Labs","Laboratories","Group","Global","Energy","Institute","Center","Centre",
           "Research","University","College","Club","Team","Athletics","United","City","Collective",
           "Ventures","Partners","Holdings","Works","Craftworks","Studio","Studios","Corp","Inc",
           "International","National","Digital","Tech","Media","Health","Bio","Capital","Network",
           "Software","Hardware","Audio","Motors","Foods","Sports","Association","Federation","Guild",
           "Games","Championship","Championships","Open","Cup","Marathon","Race","Festival","Series",
           "League","Tour","Trophy","Classic","Invitational"}

def name_tokens(fact):
    """First names of people + DISTINCTIVE base words of orgs/teams/events — generic suffixes excluded."""
    toks=set()
    for k,v in fact["entities"].items():
        if not (is_name_key(k) and isinstance(v,str)): continue
        words=[w for w in re.split(r"[\s\-]", v) if w and w[0].isupper() and w.lower() not in ("dr.","dr","prof.","prof","the","of")]
        if "person" in k.lower():
            if words: toks.add(words[0].rstrip("."))      # ban FIRST names of people
        else:
            toks.update(w for w in words if w not in GENERIC)   # only distinctive words banned
    return toks

def full_names(fact):
    return {v for k,v in fact["entities"].items() if is_name_key(k) and isinstance(v,str)}

# ===== EDITED BY CLAUDE 2026-07-19 (v2.1): domain interleaving =====
# v2.0 ran domains in blocks (all corporate, then all product, ...), so late domains (sports, mergers)
# ran when the ban list was fattest and starved (6 and 3 facts). Round-robin spreads the pressure evenly.
ORDER = [d for _ in range(20) for d in SKELETONS]   # corp, prod, sci, sport, merger, corp, prod, ...

if __name__ == "__main__":
    out=[]; used_full=set(); used_tokens=set()
    for i, dom in enumerate(ORDER):
        sk = SKELETONS[dom]
        last_clash=set()
        # ===== EDITED v2.1: 10 attempts (was 6) + escalating retry that names the clashing words =====
        for attempt in range(10):
            try:
                ban = ", ".join(sorted(used_tokens)) or "none yet"   # EDITED v2.2: FULL list (was last-80) — model must see everything it must avoid; names are short, cost is trivial
                msg = PROMPT.format(domain=dom, instruction=sk["instruction"], style=NAME_STYLES[i % len(NAME_STYLES)],
                                    ban=ban, example=json.dumps(sk["seed"], ensure_ascii=False), id=f"rb-{i:03d}")
                if last_clash:   # tell the model exactly what it reused so the retry actually changes
                    msg += (f"\n\nCRITICAL: your previous attempt reused these FORBIDDEN names: "
                            f"{', '.join(sorted(last_clash))}. Choose COMPLETELY different names.")
                r = client.chat.completions.create(model="gpt-4.1-mini", temperature=1.0,
                    response_format={"type":"json_object"}, messages=[{"role":"user","content":msg}])
                fact = extract_json(r.choices[0].message.content)
                if not has_bridge(fact): raise ValueError("no bridge")
                if full_names(fact) & used_full:
                    last_clash = full_names(fact) & used_full
                    print(f"  full-name collision at {i}: {last_clash}"); continue
                if name_tokens(fact) & used_tokens:
                    last_clash = name_tokens(fact) & used_tokens
                    print(f"  token collision at {i}: {last_clash}"); continue
                used_full |= full_names(fact); used_tokens |= name_tokens(fact)
                out.append(fact); print("ok", i, dom); break
            except Exception as e:
                print("retry", i, e); time.sleep(1)
        else:
            print(f"⚠️  gave up on fact {i} ({dom})")
    with open("data/facts.jsonl","w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} facts")

    # verify: uniqueness (full + tokens) and bridges
    tok_ct=collections.Counter(); full_ct=collections.Counter(); nb=[]
    for fct in out:
        for t in name_tokens(fct): tok_ct[t]+=1
        for n in full_names(fct): full_ct[n]+=1
        if not has_bridge(fct): nb.append(fct["id"])
    dup_t={k:c for k,c in tok_ct.items() if c>1}; dup_f={k:c for k,c in full_ct.items() if c>1}
    print("✅ full names unique" if not dup_f else f"❌ full-name dups: {dup_f}")
    print("✅ name tokens unique (no Liora problem)" if not dup_t else f"⚠️ token dups: {dup_t}")
    print("✅ all facts bridged" if not nb else f"❌ no-bridge: {nb}")
    doms=collections.Counter(f["domain"] for f in out); print("domains:", dict(doms))
    # ===== v2.1: distribution check — every domain should have >=15 facts. If a domain starved,
    # relax the first-name rule to count-based (ban only at >=2 uses) and rerun. =====
    starved=[d for d,c in doms.items() if c<15]
    print("✅ domain balance ok" if not starved else f"⚠️ starved domains (rerun or relax first-name rule): {starved}")