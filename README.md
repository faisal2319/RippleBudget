# RippleBudget

**When you inject a new fact into an LLM under a fixed data budget, does the structure of the synthetic training questions change which related questions it can then answer?**

When you fine-tune a model on a new fact, it usually learns to recall that fact but struggles to apply it to related questions. Teach it "Aiko replaced Kenji as CEO" and it answers "who is the CEO?" but fumbles "which company does Aiko lead?". This is the ripple problem.

RippleBudget tests one idea: given the same amount of training data, does generating questions that cover a fact's relationships (reverse, contradiction, one-hop) change how well the model handles those relationships later, compared to spending the same budget on more recall questions?

It builds on PASTA (Yamamoto & Kawahara, [arXiv:2606.28898](https://arxiv.org/abs/2606.28898)), which names propagation to related knowledge as a core difficulty (Section 3.1) and lists interdependent knowledge updates as future work. PASTA attacks propagation by scaling QA volume and evaluates on newly generated same-distribution questions. RippleBudget holds the token budget fixed and evaluates on held-out ripple questions the model never saw in that form.

---

## The result

Three fine-tuning conditions get the same training-token budget (matched to within one training example, 0.04%). The only thing that changes is how the questions are structured.

| Condition | Training data (matched token budget) |
|---|---|
| **A** direct-only | recall questions only |
| **B** PASTA-inspired volume baseline | direct + many paraphrased recall questions (volume, no relational structure) |
| **C** dependency-aware (ours) | direct + reverse + contradiction + one-hop questions |

Note on B: this is a volume baseline *inspired by* PASTA's finding that Context-Derived QA volume drives most of the accuracy gain. It is not a reproduction of PASTA's full CPT to SFT to DPO pipeline; it isolates the "more QA, no added structure" lever.

Ripple accuracy on held-out questions (Llama-3.2-3B, LoRA, mean ± std over 3 seeds). Contradiction is scored as **joint accuracy** (the model must both reject the stale claim and supply the correct replacement, see the breakdown below):

| Held-out class | A direct-only | B volume baseline | **C dependency-aware** |
|---|---|---|---|
| reverse | 0.240 ± 0.017 | **0.253 ± 0.009** | 0.195 ± 0.005 |
| **contradiction (joint)** | 0.034 ± 0.007 | 0.033 ± 0.018 | **0.495 ± 0.013** |
| one-hop | 0.292 ± 0.026 | **0.295 ± 0.005** | 0.243 ± 0.030 |
| compositional | 0.214 ± 0.015 | 0.239 ± 0.004 | 0.211 ± 0.007 |
| **macro-average** | 0.195 ± 0.013 | 0.205 ± 0.008 | **0.286 ± 0.013** |

What the numbers actually support:

**The result is a reallocation, not a broad improvement.** Dependency-aware QA does one thing clearly and strongly: on stale-fact contradictions it goes from near-zero joint accuracy (0.03) to about 0.50, stable across all three seeds. It does not improve propagation across the board. On reverse and one-hop, condition C is actually *worse* than the recall-based conditions. Under a fixed budget, spending data on relational coverage moves capability toward the relations you explicitly train and away from the ones you do not.

**The macro-average gap is now large and seed-stable, but read it correctly.** C's macro-average (0.286) clearly beats B (0.205) and A (0.195), and C's worst seed (0.271) sits well above B's best (0.215), so this is not noise. But almost the entire gap comes from the contradiction column. The honest reading is not "structure broadly improves propagation." It is "one relation type (contradiction) is learned very well when you train for it, enough to move the average on its own, while the others do not improve."

**The contradiction breakdown is where the real story is.** Scoring contradiction as a single number hides two very different skills. Splitting them:

| Contradiction metric | A | B | **C** |
|---|---|---|---|
| rejection (did it refuse the stale claim?) | 0.219 ± 0.024 | 0.214 ± 0.020 | **0.986 ± 0.000** |
| correction (did it give the new fact?) | 0.043 ± 0.020 | 0.057 ± 0.012 | **0.495 ± 0.013** |
| **joint (both)** | 0.033 ± 0.007 | 0.033 ± 0.018 | **0.495 ± 0.013** |

Two things worth noticing. Rejection is the easy half and C nearly saturates it (0.986, zero variance across seeds): trained on contradiction data, the model learns almost perfectly to stop affirming outdated claims. Correction is the hard half, and C only manages it about half the time, because retrieving the correct replacement value is the genuinely difficult part. For C, joint equals correction (0.495 vs 0.495), meaning whenever C produces the right new fact it has also already rejected the old one. Retrieving the correction, not rejecting the stale claim, is the real bottleneck, and that is the part still worth studying.

**Locality is roughly flat** across conditions (base 0.79 region; A 0.785, B 0.771, C 0.771), so C's contradiction gain does not come from damaging unrelated knowledge.

### What direct recall reveals: the bottleneck is entity interference, not propagation

To interpret the ripple numbers, the eval set includes a held-out direct-recall probe: one simple question per fact, testing whether the model can state the updated fact at all. Scored by correct-entity match (paraphrase-tolerant, numeric values must match exactly), mean over 3 seeds:

| Direct recall | base | A | B | C |
|---|---|---|---|---|
| held-out fact recall | 0.146 | 0.330 ± 0.018 | 0.299 ± 0.020 | 0.333 ± 0.009 |

Two things this settles. First, **the facts were acquired**: training roughly doubles direct recall over the untrained base (0.33 vs 0.15), and the base model near-floor confirms no contamination (the entities are genuinely unknown before training). Second, and more important, **acquisition is weak in a specific, diagnosable way.** Two thirds of direct questions are still answered wrong, but the errors are not refusals. Across all conditions, 94% of wrong direct answers are *confident bindings of the wrong entity* pulled from a structurally similar fact: asked who published a given paper, the model names a real-looking but incorrect researcher; asked a modem's speed, it gives a plausible wrong number. It has learned the shape of the fact and mis-binds the specific entity.

This reframes the whole result. The core difficulty here is not propagation of correctly-stored knowledge; it is **entity interference** between structurally similar updates trained together. It also explains why contradiction is the exception: a contradiction question supplies the entity in the prompt ("is the record time 2:20:00?"), so the model only has to reject and correct, sidestepping the from-scratch binding that direct, reverse, and one-hop questions all require. C's contradiction joint accuracy (0.50) sitting *above* its own direct recall (0.33) is the tell: the model handles the stale fact better when the entity is given to it than when it must retrieve it unaided.

So the honest one-paragraph summary: under a fixed budget, dependency-structured QA produces a large, seed-stable gain on stale-fact correction and no gain on other relation types, because the limiting factor is not how updates propagate but how reliably each fact's entity is bound against interference from its neighbours. That interference mechanism, not a propagation method, is the finding.

### What this pilot does and does not establish

It establishes: a clean, budget-matched comparison showing structure reallocates capability toward explicitly-trained relations (strongly for contradiction); a quantified interference failure mode (94% of direct errors are wrong-entity bindings); and that the effect is not a locality artifact. It does **not** establish that dependency-aware QA broadly improves knowledge propagation, and the absolute accuracies are low by design (3B model, no continued-pretraining stage, small fixed budget, fictional entities with a near-zero prior). The value is the controlled comparison and the mechanism it surfaces, not the absolute numbers.

---

## The controlled part

The point of the design is the fixed token budget. PASTA's limitations section notes it "did not rigorously equalize computational resources across different methods." RippleBudget does. All three conditions are packed to the same token count:

| Condition | Rows | Training tokens |
|---|---|---|
| A | 1511 | 66,725 |
| B | 1569 | 66,727 |
| C | 1440 | 68,056 |

Within 2%. So the differences in the table come from question structure, not data volume.

---

## The dataset

- **96 fictional knowledge-update facts** across 5 structurally different domains (corporate succession, product launch, scientific discovery, sports record, org merger). Fictional entities mean the base model scores near zero before fine-tuning, so any correct answer comes from training.
- Every fact has a bridge structure (a hub entity in two or more relations) so real two-hop questions are possible.
- **3,264 QA pairs** in 6 classes: direct (576), paraphrase (1152), reverse (384), contradiction (384), one_hop (384), compositional (384, held out for eval only).
- Entities are globally unique and phonetically diverse. Name origins rotate across 10 styles and distinctive names are de-duplicated, so facts do not blur into each other.

The held-out eval set is 672 ripple questions, a locality set that checks retained general knowledge and hallucination on unknown entities, and 96 held-out direct-recall probes (one per fact) used to measure acquisition.

---

## Why structure matters: the interference finding

The first version of this dataset used a single fact shape for everything ("X replaced Y as role of Org"). Error analysis showed 85% of failures were entity mis-binding: the model gave a structurally correct answer but pulled the name from a different, near-identical fact. The facts were too similar, so the model learned the template and blurred the specific names together.

Splitting the facts into 5 distinct shapes changed two things:

- Absolute ripple accuracy rose substantially. v1 sat around 0.02 to 0.18 across classes; v2 sits around 0.19 to 0.30. (Exact multipliers vary by class, so I am not going to reduce it to a single "Nx" number.)
- Mis-binding dropped from about 85% to about 45% of errors, and the errors that remained shifted from cross-fact confusion toward plausible confabulation.

That is a finding on its own: structurally similar knowledge updates interfere with each other under joint fine-tuning, and diversifying their surface structure removes a large share of that interference.

What is left is concentrated in compositional and one-hop questions, where even the best condition drops or invents a link in the multi-hop chain. Reliable multi-hop propagation is still unsolved here, and it is the direction this work points toward.

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

Note on rank: this pilot uses r=64. PASTA and Allen-Zhu & Li use r=128 and argue higher rank helps knowledge integration; r=64 was a compute choice here and the rank sensitivity was not swept.

Dataset: [`Faisal2319/ripplebudget`](https://huggingface.co/datasets/Faisal2319/ripplebudget). Related DPO/LoRA project: [`Faisal2319/readability-dpo-llama32`](https://huggingface.co/Faisal2319/readability-dpo-llama32).

---

## Limitations

- **Weak absolute acquisition.** Direct recall tops out around 0.33 even in the best condition, so the models learned the facts only partially. This is expected at 3B scale with no continued-pretraining stage and a small fixed budget, but it means the study characterizes *relative* behavior between conditions and the interference mechanism, not a system that reliably updates knowledge. Scaling acquisition (larger model, CPT stage, more budget) and re-testing whether structure still reallocates capability the same way is the natural next step.
- **Small pilot.** 96 facts, Llama-3.2-3B (PASTA used 8B), no continued-pretraining paraphrase stage. Absolute accuracy is low, which is expected at this scale. The result is the comparison across conditions, not the absolute numbers.
- **The macro-average gap is carried almost entirely by one class.** C's overall lead is real and seed-stable, but it comes from the contradiction column; on the other three classes C is flat or slightly worse. The robust finding is the contradiction reallocation (and specifically that correction, not rejection, is the bottleneck), not a broad propagation gain. Read the average with that in mind.
- **Synthetic and English-only.** Facts and QA are generated with gpt-4.1-mini and human-reviewed. Extending to Japanese knowledge-update evaluation is a natural next step.
- **LLM-judged.** Correctness is scored by an LLM judge with a small human spot-check, so some judge noise is present. Direct-recall and single-entity answers could be scored by deterministic exact match in a future version.
- **Compositional coverage depends on relational depth.** A minority of held-out compositional questions collapse to a single hop.

## Prior work

PASTA (Yamamoto & Kawahara, 2026); ripple-effect evaluation (Cohen et al., RippleEdits, 2023); reversal curse (Berglund et al., 2023); knowledge extractability through augmentation (Allen-Zhu & Li, Physics of Language Models).