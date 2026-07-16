# Federated Domain-Adaptive Tuning for Security Applications

Notebooks run on Colab (GPU) or on-prem once the cluster is available. This repo just has code, large artifacts (corpus, adapters,
checkpoints) live on Google Drive / HF and are gitignored.

## Pipeline (one job per notebook, self-contained, run in order)
| # | Notebook | Runtime | Purpose |
|---|---|---|---|
| 0 | `0_collection.ipynb` | CPU | Collect raw security corpus (ATT&CK, Sigma, NVD, CISA, CAR) |
| 1 | `1_curation.ipynb` | A100 | Clean, dedup, PII-redact, split into non-IID clients |
| 2 | `2_federated_dapt.ipynb` | L4/T4 | **Stage 1** — federated DAPT on raw text (FedAvg/FedProx/DP/Byzantine) + local-DAPT for ablation B. Saves shared adapters |
| 3 | `3_instruction_tuning.ipynb` | L4/T4 | **Stage 2 (optional)** — local per-org instruction tuning, warm-started from any Stage-1 adapter |
| 4 | `4_run_benchmarks.ipynb` | L4/T4 | Held-out eval (macro-F1 triage, multiple-choice ATT&CK). Scans all adapters → `results/*.json` |
| 5 | `5_analysis.ipynb` | **CPU** | Reads `results/*.json` → comparison table + figures. No GPU |

## Two-stage design
**DAPT is the collaboration; instruction tuning is optional local customization.**
Stage 1 (notebook 2) federates a shared domain adapter across orgs on raw private
text — the only stage that crosses org boundaries, so the only one that spends a
privacy budget. Stage 2 (notebook 3) is opt-in: each org warm-starts from a
shared adapter and tunes on its own labeled data, which never leaves. Both paths
(with / without Stage 2) are evaluated.

### The ablation that carries the paper
Every instruction-tuned row starts from a different point; only the start differs:

| Row | Starts from | Question it answers |
|---|---|---|
| A | no DAPT | Does DAPT help at all? |
| B | local DAPT (each org alone) | Does adaptation help per org? |
| C | **federated DAPT** | **Does *sharing* the DAPT help?** (headline) |
| D | centralized DAPT | Ceiling if privacy were free |

**C − B** is the measured value of joining the network — the number the whole
vision rests on.

## Data splits (train / validation / test)
`4 §1` builds a **70/15/15** split, stratified by severity for triage:
- `corpus/eval/val.json` — **validation** set, for model/hyperparameter selection.
- `corpus/eval/test.json` — **test** set, touched only for the final reported numbers.
- `corpus/eval/lm_val.json` — held-out **raw text** for DAPT perplexity/convergence.
- `corpus/eval/heldout_ids.json` — union of all three; notebooks 2 & 3 exclude these from training (no leakage).

Selection discipline: notebook 2 keeps the best federated round by **validation perplexity**;
notebook 4 scores every adapter on **both** val and test (test under canonical keys, val under
`*_val`). Pick the best config by `*_val`, then report its test numbers — never tune on test.

## How the notebooks stay decoupled
No notebook imports another. They pass **artifacts on Drive**:
- `adapters/<id>/` — LoRA weights + `meta.json` (written by 2 & 3, read by 4)
- `corpus/eval/{val,test,lm_val,heldout_ids}.json` — splits (written by 4 §1, read by 2 & 3)
- `results/<id>.json` — per-model metrics, val + test (written by 4, read by 5)

**First-run order:** `4 §1` (build splits, no GPU) → `2` → `3` (optional) → `4 §2+` → `5`.
Every notebook auto-skips finished work.

## Status
- [x] Baselines: zero-shot, local-only, centralized (see `_archive/` if kept)
- [x] Decoupled 6-notebook pipeline; eval harness fixed (macro-F1, MC ATT&CK, held-out split)
- [x] Optional Stage-2 instruction tuning + A/B/C/D ablation wired
- [x] Proper train/val/test split (70/15/15, stratified) + LM val set; validation-based round selection
- [ ] Run the Stage-1 experiment matrix on GPU (adapters not yet trained)
- [ ] Multi-seed + **validate DP accounting** (current DP-FedAvg noise is a placeholder)
- [ ] Fix NVD collection (API returned 0 CVEs — 404 in `0_collection`)
- [ ] Test & document the HuggingFace model

## Configuration (env vars / `.env`)
The notebooks are environment-portable — the bootstrap cell auto-detects Colab
vs. local/remote and reads config from environment variables, falling back to a
`.env` file in the working directory (copy `.env.example` → `.env`):

| Variable | Purpose | Default |
|---|---|---|
| `NVD_API_KEY` | NVD rate-limit key for `0_collection` | none (slower) |
| `FEDDAPT_ROOT` | where corpus/adapters/results live | Colab: Drive; else `./FedDAPT` |
| `FEDDAPT_WORK` | fast scratch dir for training temp | system temp dir |

Real environment variables override `.env`. `.env` is gitignored; `.env.example` is the template.

## Running in PyCharm on a remote GPU
You edit locally in PyCharm; the kernel runs on the GPU box (on-prem cluster or a
cloud instance). Two supported paths:

**A. Remote Jupyter kernel (simplest)**
1. On the GPU box: `pip install jupyter` then
   `jupyter notebook --no-browser --port=8888 --NotebookApp.token=<token>`
   (or `--ServerApp.token`). If it's remote, tunnel it: `ssh -L 8888:localhost:8888 user@gpubox`.
2. In PyCharm: open a notebook → the **Jupyter server** dropdown (top of the notebook) →
   **Configure Jupyter Server** → *Configured Server* → `http://localhost:8888/?token=<token>`.
3. Set config on the GPU box before launching Jupyter:
   `export FEDDAPT_ROOT=/data/fedapt` (and `NVD_API_KEY=...`), or drop a `.env` in the run dir.

**B. SSH interpreter (PyCharm Professional)**
*Settings → Project → Python Interpreter → Add → SSH*, point at the box's Python/conda env.
PyCharm runs notebook cells through that remote interpreter; set the same env vars there.

Notes: the `!pip` / `!git` / `!nvidia-smi` cells work in any Jupyter kernel, including
remote. The `FEDDAPT_ROOT` you set on the remote box is where all artifacts are read/written,
so points 2–5 stay in sync across notebooks without touching Drive.

## Working setup summary
- **GPU execution** → remote kernel/SSH interpreter (above), or Colab (*Open notebook → GitHub tab* → *Save a copy in GitHub*).
- **Local editing** → PyCharm Professional / DataSpell (free with a `.edu` student license).
- **Big data** → remote box / Google Drive / HF, gitignored here.
