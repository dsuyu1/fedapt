# FeDAPT

Federated domain-adaptive pre-training for **cross-domain security assistant
LLMs** — can a network of orgs collaboratively pre-train a better security
assistant without sharing private data? Full rationale in **[DESIGN.md](DESIGN.md)**.

## Idea in one line
**Prose** (public security text) is the shared knowledge → federated DAPT.
**Logs** (private telemetry) are each org's grounding → local task tuning.
DAPT is the collaboration; task tuning is optional local specialization.

## Layout
```
src/fedapt/          # the library — one idea per module
  config.py            all knobs, seeds, paths (start here)
  corpus.py            build the PROSE DAPT corpus            [data layer ✓]
  vendor_feeds.py      harvest vendor threat-intel via RSS    [✓]
  clients.py           non-IID client split (prose + logs)    [data layer ✓]
  tasks.py             the 4 task builders + teacher targets   [data layer ✓]
  splits.py            train/val/test + held-out ids           [data layer ✓]
  model.py             base + LoRA (QLoRA) plumbing            [✓]
  aggregators.py       fedavg / krum / trimmed_mean           [✓]
  attacks.py           malicious-client simulation            [✓]
  dp.py                DP-FedAvg (placeholder accountant)      [✓]
  eval_metrics.py      ROUGE / BERTScore (secondary)          [✓]
  judge.py             CLEV LLM-as-judge (primary)            [✓]
  federated.py         Stage-1 federated DAPT + local DAPT    [✓]
  train_tasks.py       Stage-2 per-org task tuning (A/B/C)    [✓]
  evaluate.py          eval driver: Macro-F1 + CLEV judge     [✓]
  analysis.py          table + figures from results/ (no GPU) [✓]
scripts/               # CLI drivers: build_data / train / evaluate / analyze
notebooks/             # thin Colab drivers 00-04 (install -> load_config -> call driver)
tests/                 # fast unit tests (16, all pure-Python)
legacy/                # the old monolithic notebooks, for reference only
```

**Full end-to-end run: see [RUNBOOK.md](RUNBOOK.md).**

## Quickstart
```bash
pip install -e .                 # core (data layer). Extras when needed:
pip install -e ".[train]"        # GPU training (torch/transformers/peft/bnb)
pip install -e ".[eval]"         # metrics + LLM judge

cp .env.example .env             # set NVD_API_KEY, FEDDAPT_ROOT, FEDDAPT_ATTACK_DATA

python scripts/build_data.py                        # offline (weak targets)
python scripts/build_data.py --teacher gpt-4o-mini  # real synthesized targets
pytest                                              # data-layer tests
```

## The four tasks (output is always prose)
1. **explain_example** — walk through a scenario  ·  2. **explain_log** — interpret telemetry *(primary)*
3. **general_qa** — open question  ·  4. **verdict** — "is this bad, and why," paragraph form *(primary)*

## Evaluation
Split 70/15/15 (stratified); pick by **val**, report **test** once.
ROUGE/BERTScore are secondary; the primary correctness signal is a **CLEV
LLM-as-judge** (verdict + rationale, 2+1 voting) — with an **independent** judge
model, **validated against a human subset** (Cohen's κ + macro-F1), at temp 0.

## Config / secrets
`.env` (gitignored) or env vars, read by `fedapt.config`:
`NVD_API_KEY`, `FEDDAPT_ROOT`, `FEDDAPT_WORK`, `FEDDAPT_ATTACK_DATA`,
`FEDDAPT_JUDGE_MODEL`, `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`).

Log client data is the public **`splunk/attack_data`** repo (git-lfs, ~9 GB) —
clone it separately and point `FEDDAPT_ATTACK_DATA` at the checkout.

## Status
- [x] DESIGN.md; clean `src/` package; data layer (corpus/clients/tasks/splits) + tests
- [x] Aggregators, attacks, DP, metrics, CLEV judge implemented
- [x] Stage-1 federated DAPT loop (`federated.py`) + experiment matrix driver
- [x] Local DAPT (ablation B) + Stage-2 per-org task tuning (`train_tasks.py`)
- [x] Evaluation driver (`evaluate.py`): verdict Macro-F1 + ROUGE + CLEV judge; `scripts/train.py`, `scripts/evaluate.py`
- [x] Analysis (`analysis.py` + `scripts/analyze.py`): comparison table + ablation / privacy-utility / Byzantine / learning-curve figures
- [x] Benign log source + class-balanced verdict task (real dir via `FEDDAPT_BENIGN_DATA`, or teacher-synthesised negatives)
- [x] Real DP accounting (Opacus RDP): target ε → noise σ, and the ε actually spent recorded per run
- [x] Thin Colab notebooks (`notebooks/00–04`) wrapping every driver
- [x] Vendor threat-intel RSS harvester (`scripts/fetch_vendor_rss.py`) → prose corpus
- [ ] Obtain real benign captures; run the full matrix on GPU; validate the judge vs a human subset
- [ ] Real DP accountant; benign log source for the verdict task
