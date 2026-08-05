# FeDAPT — Runbook

Exact end-to-end sequence for a full run. Two paths: **local/remote GPU** (CLI)
or **Colab** (notebooks). Everything is resumable — finished adapters/results are
skipped, so you can stop and resume any stage.

---

## 0. Prerequisites
- Python ≥ 3.10. A CUDA GPU for the training/eval stages (Stage 1/2, eval); the
  data build and analysis are CPU-only. **No local GPU? Use the Colab path below.**
- `git` + `git-lfs` (only if you use `attack_data`; the paper-backed datasets don't need it).
- For target synthesis + the judge: **either** an LLM API key **or** a local
  **Ollama** server — using models **different** from the Mistral-7B base. The
  local path needs no key and can't be rate-limited (see step 4).

## 1. Get the code + install
```bash
git clone https://github.com/YOUR_USERNAME/fedapt.git
cd fedapt
pip install -e ".[train,eval]"        # CPU-only? use: pip install -e .
pytest                                # 31 tests, ~seconds — sanity check
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
Both classes for the verdict task (recommended): use paper-backed public datasets
via the normalized log-source layer — no Splunk lab needed. Convert once, then
point `FEDDAPT_LOG_SOURCES` at the output (see `docs/BENIGN_DATA.md`):
```bash
# AIT (endpoint+network, both classes; Zenodo 10.5281/zenodo.5789064)
python scripts/convert_dataset.py --dataset ait --src /data/AIT/<dataset> --out data/normalized/ait.jsonl
# CIC-IDS2017 / UNSW-NB15 (network) · CloudTrail (cloud, one --label per export)
```
If you skip this, set a teacher (step 3) and negatives are synthesised (disclosed).

## 3. Configure
```bash
cp .env.example .env
```
Edit `.env`:
```
FEDDAPT_ROOT=/data/fedapt                 # where corpus/adapters/results live
FEDDAPT_ATTACK_DATA=/abs/path/to/attack_data
FEDDAPT_BENIGN_DATA=/abs/path/to/benign   # optional (raw benign .log/.json dir)
FEDDAPT_LOG_SOURCES=data/normalized       # optional: normalized dataset .jsonl (both classes)
NVD_API_KEY=...                           # optional (faster CVE collection)
FEDDAPT_JUDGE_MODELS=claude-haiku-4-5,claude-sonnet-4-6   # API judges; two -> CLEV voting; judge ≠ base
# or LOCAL (no key):  FEDDAPT_JUDGE_MODELS=ollama:gemma3:12b,ollama:llama3.1:8b
ANTHROPIC_API_KEY=...                     # only if using API models
FEDDAPT_OLLAMA_HOST=http://localhost:11434  # only if using ollama:/local: models
```

## 3a. Where everything is stored (read this — it's the #1 confusion)
**One knob controls it: `FEDDAPT_ROOT`.** Corpus, clients, tasks, adapters, results,
and figures all live under it — nothing is written anywhere else, and Google Drive is
*not* special. It only ever enters the picture because of what `FEDDAPT_ROOT` resolves to:

- **Local run:** `FEDDAPT_ROOT` = whatever you set in `.env` (e.g. `/data/fedapt`), else
  `./FedDAPT` next to the repo. Results stay **on that machine**. They do **not** appear
  in Google Drive unless that path is itself inside a Drive-synced folder.
- **Colab run:** if `FEDDAPT_ROOT` is blank, `Config` mounts Google Drive and uses
  `/content/drive/MyDrive/FedDAPT`. So on Colab, results persist in **your Drive** and
  survive session drops.

There is **no automatic sync** between local and Colab. A given run has exactly one home.
To move a run between machines, copy the `FEDDAPT_ROOT` folder (Drive is the easy bridge).
The notebooks and the `scripts/` CLI are just two front-ends to the *same* functions and
write to the *same* `FEDDAPT_ROOT` — use whichever matches where your GPU is.

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
python scripts/build_data.py                          # offline metadata targets
# python scripts/build_data.py --teacher claude-haiku-4-5   # API teacher
# python scripts/build_data.py --teacher ollama:gemma2:9b   # LOCAL teacher (no key, no rate limits)
```
No API budget? Run a **local** teacher via Ollama — free, private, and immune to the
rate-limit fallbacks that can silently poison targets. Install Ollama, `ollama pull
gemma2:9b` (or `qwen2.5:7b` / `llama3.1:8b`), then `--teacher ollama:<model>`. It's
slower per call but ~2k targets finish overnight; watch the `⚠ fell back` line in the
output — it should read 0. Disclose the teacher model in the paper.
Produces under `$FEDDAPT_ROOT`: `corpus/`, `clients/`, `tasks/`,
`eval/{*_split.json, lm_val.json, heldout_ids.json}`.

## 4c. Smoke test (GPU) — do this before the full run
Runs the entire path (federated DAPT → task tuning → eval → analysis) on a tiny
config in a few minutes, so integration bugs surface cheaply. Isolated under a
`smoke` run_name — never touches a real run.
```bash
pip install -e ".[train,eval]"
python scripts/smoke.py
```
If it prints "✅ SMOKE PASSED" and a comparison table, the pipeline works end to
end. Artifacts land in `$FEDDAPT_ROOT/{adapters,results,figures}/smoke` — delete
those folders before the real run if you like.

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
Generate a 50-item audit sheet from the student model, hand-label it, then score
the judge against your labels (see `docs/JUDGE_VALIDATION.md` for the full walkthrough):
```bash
python scripts/make_judge_audit.py --n 50 --mix dapt_fedavg_no_dp   # GPU; writes eval/judge_audit.csv
#   ↳ fill the `human` column (1/0) by hand — free, no compute, label BLIND
python scripts/score_judge.py                                       # prints Cohen's κ + macro-F1
```
Judge models come from `FEDDAPT_JUDGE_MODELS` (API `claude-…` or local `ollama:…`).
Only trust the judge if it clears the CLEV bar (κ≥0.6, macro-F1≥0.85); otherwise
pick a stronger judge model and re-score.

---

## Colab GPU path (offload the GPU stages)
Goal: keep the CPU work (data build) local, and run the GPU-heavy stages
(smoke → train → evaluate) on Colab's GPU. Everything reads one `Config`, and on
Colab `FEDDAPT_ROOT` **auto-defaults to `/content/drive/MyDrive/FedDAPT`** — so if
your data lives in that Drive folder, Colab picks it up with zero config.

**Recommended split.**
1. **Local (overnight, CPU):** run steps 1–4 to build the data with the Ollama
   teacher. Point `FEDDAPT_ROOT` at a Google-Drive-synced folder (or zip
   `$FEDDAPT_ROOT` and upload it to `MyDrive/FedDAPT`). You want `corpus/`,
   `clients/`, `tasks/`, and `eval/` sitting in Drive.
2. **Colab (GPU):** Runtime → Change runtime type → **GPU** (T4 is enough; A100 if
   you have Pro). Then in a cell:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !git clone https://github.com/YOU/fedapt.git
   %cd fedapt
   !pip install -e ".[train,eval]" -q
   # FEDDAPT_ROOT auto-resolves to /content/drive/MyDrive/FedDAPT in Colab
   !python scripts/smoke.py                 # 2-round sanity check first
   !python scripts/train.py --all           # local DAPT → federated matrix → task tuning
   !python scripts/evaluate.py              # verdict Macro-F1 + ROUGE
   ```
   Or just open `notebooks/00_build_data → 04_analysis` and run top-to-bottom —
   they wrap these same scripts and are pre-sliced to fit a session.

**Colab gotchas that actually bite:**
- **Sessions drop** (idle / wall-clock limits). Harmless here: every adapter/result is
  written to Drive and finished ones auto-skip, so just **reconnect and re-run** to
  resume. Run the 17-experiment matrix in slices across sessions.
- **GPU memory:** Mistral-7B in 4-bit needs ~6–8 GB; a free T4 (16 GB) handles one
  experiment at a time. Don't try to parallelize on one GPU.
- **Secrets:** put `ANTHROPIC_API_KEY` etc. in Colab **Secrets** (🔑 sidebar) or a
  `.env` in the repo — `Config` reads env vars, `.env`, and Colab `userdata`.
- **Judge on Colab:** the CLEV judge (`evaluate.py --judge`) wants an API key or a
  reachable Ollama host. Simplest is to run plain `evaluate.py` on Colab (Macro-F1 +
  ROUGE) and do the `--judge` pass separately — locally with Ollama, or a short API
  session — so the judge doesn't compete with training for the GPU.
- **Don't run the Ollama teacher on Colab:** it would fight training for the GPU.
  Build data locally (step 4); use Colab's GPU purely for training/eval.

## Reproducibility notes
- All splits/partitions are seeded (`Config.seed`); re-running a stage is deterministic.
- One Config fully describes a run — change knobs there, not in scattered cells.
- Change the corpus? Re-run step 4 so `heldout_ids` stay in sync (no leakage).
