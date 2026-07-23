# FeDAPT — Design Document

Clean-slate design. The old notebooks are preserved in `legacy/` for reference;
nothing here depends on them. Goal: one clear story, one clean codebase,
explainable end to end.

---

## 1. Research question

**Can a network of security organizations collaboratively pre-train a better
domain assistant LLM — one that generalizes and specializes across security
domains — without sharing their private data?**

We are **not** building an anomaly/attack detector. We are building an
**assistant** that reasons about security in natural language, and testing
whether *federated* domain-adaptive pre-training beats each org going it alone,
while keeping private data local and staying robust to privacy noise and
malicious participants.

---

## 2. Design principles (what "clean" means here)

1. **One idea per module.** Small, named, testable Python modules; notebooks are
   thin drivers that call them. No 300-line self-contained cells.
2. **Reproducible.** Public data + fixed seeds. Every result regenerable from a
   config. No hidden state.
3. **Honest.** Placeholders are labeled as such (DP accounting), simulated
   settings are called simulated (public data as stand-in for private).
4. **Decoupled by artifacts.** Stages read/write files on disk; any stage runs
   alone.

---

## 3. Data model — the key distinction

| Layer | What it is | Role | Privacy |
|---|---|---|---|
| **DAPT corpus** | **Prose**: ATT&CK/D3FEND text, Sigma/CVE/CISA descriptions, threat-intel & IR writeups (DFIR Report), CIS/NIST docs, Atomic Red Team *descriptions*, security Q&A | Shared knowledge substrate; teaches security *language & reasoning* | Public / shareable-in-simulation → the **federated** part |
| **Client data** | **Datasets (logs)**: `splunk/attack_data` telemetry (sysmon, 4688, cloudtrail…) with MITRE technique + benign/malicious labels | Per-org grounding; the model learns to *interpret* an org's telemetry | Private in the story (public stand-in here) → the **local** part |

**Why this split works:** prose generalizes and is safe to pool → good for
federated DAPT. Logs are the genuinely sensitive, org-specific data → they stay
local and only ever appear as *task inputs*, never in the shared corpus. Logs
partition naturally by domain (sysmon=endpoint, zeek/firewall=network,
cloudtrail=cloud), giving **real non-IID** client structure.

> **Public data is deliberate**, not a shortcut. It buys reproducibility and
> comparability. `attack_data` is itself lab-generated org telemetry, so we get
> "simulated organizations" without running cloud labs. Standing up real labs /
> expert-designed orgs is **v2 / future work**, not needed to prove the method.

---

## 4. Two-stage training

```
Stage 1 — FEDERATED DAPT (shared, collaborative, private-budget spent here)
    each org pre-trains on the PROSE corpus (non-IID shards)
    LoRA updates aggregated across orgs  ->  shared domain adapter
    this is the only stage where anything crosses org boundaries

Stage 2 — LOCAL TASK TUNING (optional, per org, data never leaves)
    warm-start from the shared adapter
    tune on the org's own (log -> prose) task pairs
    produces an org-specialized assistant
```

DAPT is the collaboration; task tuning is optional local specialization. Both
paths (with / without Stage 2) are evaluated.

### The ablation that carries the paper
Same task tuning, different starting point — only the start differs:

| Row | Start from | Question |
|---|---|---|
| A | no DAPT | Does DAPT help at all? |
| B | local DAPT (org alone) | Does adaptation help per org? |
| C | **federated DAPT** | **Does *sharing* help?** (headline) |
| D | centralized DAPT | Ceiling if privacy were free |

**C − B** = the measured value of joining the network.

---

## 5. Federated mechanics (kept, but clean)

- **Aggregation:** FedAvg + FedProx (drift control for non-IID).
- **Privacy:** DP-FedAvg (clip + Gaussian noise) with a **real RDP accountant**
  (Opacus): user-level DP, full client participation, composed over `num_rounds`.
  Target ε → noise σ; the ε *actually spent* is recorded per run for honest
  reporting. (Offline without Opacus it falls back to a crude map, flagged as
  uncertified.)
- **Byzantine / malicious clients:** 6 clients (Dirichlet non-IID); 1 malicious.
  Attacks: `sign_flip`, `scale`, `gaussian` (update-level) and `data_poison`
  (train on scrambled logs). Robust aggregators: Krum, trimmed-mean. Story:
  under attack FedAvg degrades, robust aggregators hold. (n≥5 needed for Krum.)

---

## 6. Task taxonomy (the four questions)

Model input varies; **output is always prose**. Labels come from dataset
metadata + teacher-synthesized targets (see §7).

1. **Explain an example** — scenario/case walk-through. (src: ATT&CK procedures, DFIR)
2. **Explain a log snippet** — interpret telemetry. *The org-private, most novel task.* (src: `attack_data`)
3. **General security question** — open QA. (src: security Q&A, synthesized from prose)
4. **Verdict with rationale** — "is this bad, and why," in paragraph form. (src: `attack_data` benign/malicious labels)

Tasks 2 and 4 are the **primary contribution**; 1 and 3 are auxiliary. Task 4
paragraph-form both matches how analysts communicate and avoids the brittle
exact-match metric that sank the old ATT&CK eval.

---

## 7. Label provenance (how log tasks get targets)

A raw log has no natural-language target. We build them:

- **Hard labels** (task 4): straight from `attack_data` YAML — technique id,
  malicious vs benign. Real ground truth.
- **Prose targets** (tasks 1–3, and task-4 rationale): **teacher-synthesized** —
  a strong model writes the explanation from the log + its technique label +
  description. Disclosed as distillation; a small subset human-checked.
- **Private framing:** in deployment these targets would be an org's own analyst
  notes; here `attack_data` is the public stand-in. Stated explicitly.

---

## 8. Evaluation

**Split discipline:** 70/15/15 train/val/test, stratified. Pick configs by
**validation**; report **test** once. Never tune on test. Held-out raw-text set
for DAPT perplexity/convergence.

**Metrics, layered:**
- **Secondary (cheap, reference-based):** ROUGE-L + BERTScore. Reported, but we
  note (per Badshah et al., CLEV) they undercount semantic equivalence.
- **Primary (free-form correctness):** **LLM-as-a-judge, CLEV-style** —
  reference-aware, returns a **binary verdict + rationale**; two judges vote, a
  third breaks ties (~80–95% cheaper than a 3-judge panel).
- **Task 4 also gets Macro-F1** by parsing the model's own verdict against the
  dataset's benign/malicious ground truth → one hard number *plus* judged rationale.

**Judge integrity (non-negotiable for review):**
- **Independent judge model** — NOT the base model under test (avoid
  self-preference bias). Set via env var.
- **Validate the judge** against a small human-annotated subset via Cohen's κ +
  Macro-F1 (CLEV admits judges only above a bar). This is what makes the judge defensible.
- Judge at **temperature 0** + voting for run-to-run consistency.

---

## 9. Proposed repo structure (clean)

```
fedapt/
  DESIGN.md                 # this file
  README.md                 # quickstart + pointers
  pyproject.toml            # pip install -e .   (works on Colab/remote)
  .env / .env.example       # secrets + FEDDAPT_ROOT
  src/fedapt/
    config.py               # one dataclass config; all knobs, seeds
    corpus.py               # build the PROSE DAPT corpus
    clients.py              # build log CLIENT datasets + non-IID split
    tasks.py                # the 4 task builders + teacher target synthesis
    splits.py               # train/val/test + held-out ids
    model.py                # load base + LoRA (QLoRA)
    federated.py            # FedAvg/FedProx loop, round selection
    aggregators.py          # fedavg / krum / trimmed_mean
    attacks.py              # sign_flip / scale / gaussian / data_poison
    dp.py                   # DP-FedAvg (+ TODO real accountant)
    eval_metrics.py         # ROUGE / BERTScore
    judge.py                # CLEV LLM-as-judge + human-validation harness
    train_dapt.py           # Stage 1 driver
    train_tasks.py          # Stage 2 driver
  notebooks/                # THIN drivers that call src/ (Colab-friendly)
    00_build_data.ipynb
    01_federated_dapt.ipynb
    02_task_tuning.ipynb
    03_evaluate.ipynb
    04_analysis.ipynb
  legacy/                   # old notebooks, for reference only
```

Modules are unit-testable and reliable to author as `.py`; notebooks stay short.

---

## 10. What we scrap / keep

- **Scrap:** the 6 monolithic self-contained notebooks (duplicated helpers,
  giant cells, hard to explain). Preserved in `legacy/`.
- **Keep (reimplement cleanly):** the good ideas already worked out — two-stage
  design, A/B/C/D ablation, non-IID split, DP-FedAvg, Byzantine attacks +
  robust aggregators, train/val/test discipline, portable `.env` bootstrap.

---

## 11. Open decisions / risks

- [ ] Code structure: `src/` package + thin notebooks (recommended) vs. fewer clean notebooks.
- [ ] Which teacher model synthesizes prose targets; how large a human-checked subset.
- [ ] Which independent judge model (and its cost/availability on your setup).
- [ ] Obtain real benign captures (else the verdict negatives are teacher-synthesised, disclosed).
- [ ] Scope guard: anchor v1 on tasks 2 + 4; keep 1 + 3 light.

*(Resolved: DP now uses a real Opacus RDP accountant; the verdict task has a
benign class via `FEDDAPT_BENIGN_DATA` or synthesised negatives.)*
