# FeDAPT — Runbook

Exact end-to-end sequence for a full run. Two paths: **local/remote GPU** (CLI)
or **Colab** (notebooks). Everything is resumable — finished adapters/results are
skipped, so you can stop and resume any stage.

---

## 0. Prerequisites
- Python ≥ 3.10. A CUDA GPU for the training/eval stages (Stage 1/2, eval); the
  data build and analysis are CPU-only.
- `git` + `git-lfs` (for `attack_data`).
- An LLM API key for the judge / teacher (a model **different** from the base under test).

## 1. Get the code + install
```bash
git clone https://github.com/YOUR_USERNAME/fedapt.git
cd fedapt
pip install -e ".[train,eval]"        # CPU-only? use: pip install -e .
pytest                                # 16 tests, ~seconds — sanity check
```

## 2. Get the client log data
`attack_data` is git-lfs (~9 GB) — pull a subset first, expand later.
```bash
git clone https://github.com/splunk/attack_data.git ../attack_data
cd ../attack_data && git lfs install --skip-smudge
# pull a handful of techniques to start (add more folders anytime):
git lfs pull --include="datasets/attack_techniques/T1003*/**"
git lfs pull --include="datasets/attack_techniques/T1059*/**"
cd ../fedapt
```
Benign negatives (recommended for a real verdict metric): drop a directory of
benign `.log`/`.json` captures somewhere (OTRF Security-Datasets baseline runs,
an attack_range benign run, or a public benign network dataset). If you skip
this, set a teacher (step 3) and negatives are synthesised (disclosed).

## 3. Configure
```bash
cp .env.example .env
```
Edit `.env`:
```
FEDDAPT_ROOT=/data/fedapt                 # where corpus/adapters/results live
FEDDAPT_ATTACK_DATA=/abs/path/to/attack_data
FEDDAPT_BENIGN_DATA=/abs/path/to/benign   # optional
NVD_API_KEY=...                           # optional (faster CVE collection)
FEDDAPT_JUDGE_MODELS=claude-3-5-haiku-20241022,claude-3-5-sonnet-20241022   # two -> CLEV voting; judge ≠ base model
ANTHROPIC_API_KEY=...
```

## 3b. (Optional) Harvest vendor threat-intel prose  (CPU)
Enriches the DAPT corpus with public IR/threat-intel write-ups via official RSS feeds.
```bash
pip install -e ".[fetch]"
python scripts/fetch_vendor_rss.py --out ./vendor_articles           # feed summaries
# python scripts/fetch_vendor_rss.py --out ./vendor_articles --full  # full article text (slower, robots-respecting)
```
Then set `FEDDAPT_VENDOR_DATA=/abs/path/to/vendor_articles` in `.env`. Re-runnable
(incremental); edit the feed list in `fedapt.vendor_feeds.DEFAULT_FEEDS` or pass
`--feeds my_feeds.json`. Respect each source's terms.

## 4. Build the data  (CPU)
```bash
python scripts/build_data.py                        # offline metadata targets
# python scripts/build_data.py --teacher gpt-4o-mini  # real synthesised targets + benign
```
Produces under `$FEDDAPT_ROOT`: `corpus/`, `clients/`, `tasks/`,
`eval/{*_split.json, lm_val.json, heldout_ids.json}`.

## 5. Train  (GPU)
```bash
python scripts/train.py --local     # local DAPT per org        (ablation B)
python scripts/train.py --stage1    # federated DAPT matrix     (Stage 1)
python scripts/train.py --stage2    # task tuning A/B/C          (Stage 2)
# or all in order:
python scripts/train.py --all
```
The matrix is 17 experiments; each reloads a fresh 7B base, so on a single GPU
run it in slices. To limit it, edit `federated.default_matrix` or drive from the
notebook (`run_matrix(cfg, specs[:k])`). All runs auto-skip if their adapter exists.

## 6. Evaluate  (GPU)
```bash
python scripts/evaluate.py            # verdict Macro-F1 + ROUGE (no judge)
python scripts/evaluate.py --judge    # + CLEV LLM-judge (needs the keys above)
```
Writes `results/<id>.json` (test metrics under canonical keys, val under `*_val`).

## 7. Analyse  (CPU)
```bash
python scripts/analyze.py
```
Writes `figures/`: `comparison_table.csv`, `ablation.png`, `privacy_utility.png`,
`byzantine.png`, `learning_curves.png`. The ablation step prints **C − B** — the
value of federating.

## 8. Validate the judge (do this before trusting judge numbers)
Hand-label a small set (~50) of (input, candidate, reference, human_correct) and:
```python
from fedapt.config import load_config
from fedapt.judge import make_llm, validate_judge
cfg = load_config()
judges = [make_llm(m) for m in ["gpt-4o-mini", "gpt-4o"]]
print(validate_judge(judges, my_labeled_examples))   # Cohen's kappa + macro-F1
```
Report κ and macro-F1; only trust the judge if it clears the bar (CLEV: κ≥0.6, F1≥0.85).

---

## Colab path (instead of steps 4–7)
Open `notebooks/00_build_data → 04_analysis` from the GitHub tab. Set your repo
URL in the first cell, put the same values in `.env` (or Colab Secrets), and run
top to bottom. `01` is sliced so it fits a session; re-open and it resumes.

## Reproducibility notes
- All splits/partitions are seeded (`Config.seed`); re-running a stage is deterministic.
- One Config fully describes a run — change knobs there, not in scattered cells.
- Change the corpus? Re-run step 4 so `heldout_ids` stay in sync (no leakage).
