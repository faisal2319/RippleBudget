# RippleBudget

**Does the structure of synthetic training questions matter more than their quantity when you inject a new fact into an LLM?**

When you fine-tune a model on a new fact, it usually learns to recall that fact but fails to apply it to related questions. Teach it "Aiko replaced Kenji as CEO" and it answers "who is the CEO?" but fumbles "which company does Aiko lead?". This is the ripple problem.

RippleBudget tests one idea: given the same amount of training data, does generating questions that cover a fact's relationships (reverse, contradiction, one-hop) help the model apply that fact better than just generating more recall questions?

It builds on PASTA (Yamamoto & Kawahara, [arXiv:2606.28898](https://arxiv.org/abs/2606.28898)), which names propagation to related knowledge as a core difficulty and lists interdependent knowledge updates as future work. PASTA attacks propagation by scaling QA volume and tests on same-distribution questions. RippleBudget holds volume fixed and tests on held-out ripple questions the model never saw in that form.

---

## The result

Three fine-tuning conditions get the exact same training-token budget. The only thing that changes is how the questions are structured.

| Condition | Training data (equal tokens) |
|---|---|
| **A** direct-only | recall questions only |
| **B** PASTA-style | direct + lots of paraphrased recall questions (volume, no structure) |
| **C** dependency-aware (ours) | direct + reverse + contradiction + one-hop questions |

Ripple accuracy on held-out questions (Llama-3.2-3B, LoRA, mean over 3 seeds):

| Held-out class | A direct-only | B PASTA-style | **C dependency-aware** |
|---|---|---|---|
| reverse | 0.240 | **0.253** | 0.194 |
| **contradiction** | 0.059 | 0.049 | **0.253** |
| one-hop | 0.292 | **0.295** | 0.243 |
| compositional | 0.214 | 0.240 | 0.211 |
| **overall (avg)** | 0.201 | 0.209 | **0.225** |

Two things stand out.

First, dependency-aware QA gives roughly a 5x gain on stale-fact contradiction (0.253 vs 0.049 and 0.059). That is the one capability it directly trains, and it also gives the best overall ripple average, all at equal budget.

Second, it is a trade-off, not a clean sweep. C trails A and B slightly on reverse and one-hop, because under a fixed budget it spends data on relational coverage that the others spend on recall volume. Structure buys a large, specific capability (rejecting outdated facts) at a small cost to breadth.

Locality stays flat across all three conditions (A 0.785, B 0.771, C 0.771), so C's gain does not come from damaging unrelated knowledge.

---

## The controlled part

The whole point is the fixed token budget. PASTA's own limitations section says it "did not rigorously equalize computational resources across different methods." RippleBudget does. All three conditions are packed to the same token count:

| Condition | Rows | Training tokens |
|---|---|---|
| A | 1511 | 66,725 |
| B | 1569 | 66,727 |
| C | 1440 | 68,056 |

Within about 2%. So any difference in the table above comes from question structure, not data volume.

---

## The dataset

- **96 fictional knowledge-update facts** across 5 structurally different domains (corporate succession, product launch, scientific discovery, sports record, org merger). Fictional entities mean the base model scores near zero before fine-tuning, so any correct answer comes from training.
- Every fact has a bridge structure (a hub entity in two or more relations) so real two-hop questions are possible.
- **3,264 QA pairs** in 6 classes: direct (576), paraphrase (1152), reverse (384), contradiction (384), one_hop (384), compositional (384, held out for eval only).
- Entities are globally unique and phonetically diverse. Name origins rotate across 10 styles and distinctive names are de-duplicated, so facts do not blur into each other.

The held-out eval set is 672 ripple questions plus a locality set that checks retained general knowledge and hallucination on unknown entities.

---

## Why structure matters: the interference finding

The first version of this dataset used a single fact shape for everything ("X replaced Y as role of Org"). Error analysis showed 85% of failures were entity mis-binding: the model gave a structurally correct answer but pulled the name from a different, near-identical fact. The facts were too similar, so the model learned the template and mushed the specific names together.

Splitting the facts into 5 distinct shapes changed two things:

- Absolute ripple accuracy went up about 10x (from 0.02 to 0.18 in v1, up to 0.19 to 0.30 in v2).
- Mis-binding dropped from about 85% to about 45% of errors, and the errors that remained shifted from cross-fact confusion toward plausible confabulation.

That is a finding on its own: structurally similar knowledge updates interfere with each other under joint fine-tuning, and diversifying them fixes a lot of it.

What is left is concentrated in compositional and one-hop questions. Even the best condition drops or makes up a link in the multi-hop chains. Getting multi-hop propagation to work reliably is still unsolved, and it is the direction this work points toward.

---

## Reproduce

```bash
# 1. build the data (laptop, OpenAI API)
python src/1_generate_facts.py        # 96 fictional bridge-structured facts
python src/3_generate_qa.py           # 6-class QA
python src/4_build_conditions.py      # budget-matched train_A/B/C + eval_ripple
python src/build_locality.py          # locality probes

# 2. train + generate (GPU)
python src/2_contamination_check.py   # check base model doesn't already know the entities
for C in A B C; do for S in 0 1 2; do python src/5_train.py $C $S; done; done
python src/6_generate_answers.py      # answers on held-out eval

# 3. score (laptop, OpenAI judge)
python src/6_judge.py                 # per-class ripple table + locality
```

Setup: Llama-3.2-3B-Instruct, LoRA (r=64), 3 epochs, 3 seeds per condition. Single RTX 4090, about 3.5 min per run. Fact and QA generation plus judging use gpt-4.1-mini.

Dataset: [`Faisal2319/ripplebudget`](https://huggingface.co/datasets/Faisal2319/ripplebudget). Related DPO/LoRA project: [`Faisal2319/readability-dpo-llama32`](https://huggingface.co/Faisal2319/readability-dpo-llama32).

---

## Limitations

- Small pilot. 96 facts, Llama-3.2-3B (PASTA used 8B), no continued-pretraining paraphrase stage. Absolute accuracy is low, which is expected given the scale and the difficulty of propagation. The result is the comparison across conditions, not the absolute numbers.
- Synthetic and English-only. Facts and QA are generated with gpt-4.1-mini and human-reviewed. Extending to Japanese knowledge-update evaluation is a natural next step.
- Correctness is scored by an LLM judge with a small human spot-check, so some judge noise is present.
- Compositional coverage depends on how much relational depth a fact has. A minority of held-out compositional questions collapse to a single hop.

## Prior work

PASTA (Yamamoto & Kawahara, 2026); ripple-effect evaluation (Cohen et al., RippleEdits, 2023); reversal curse (Berglund et al., 2023); knowledge extractability through augmentation (Allen-Zhu & Li, Physics of Language Models).