# FeDAPT: Federated Domain-Adaptive Tuning for Security Applications
 
I run the .IPYNB notebooks on Colab (GPU) (or on-prem once I have access to the cluster).
 
## Structure
| Notebook | Runtime | Purpose |
|---|---|---|
| `notebooks/01_collection.ipynb` | CPU | Collect raw security corpus (ATT&CK, Sigma, NVD, CISA, CAR) |
| `notebooks/02_curation.ipynb` | A100 | Clean, dedup, PII-redact, split into non-IID clients |
| `notebooks/03_train_eval.ipynb` | L4/T4 | Baselines + 12 federated experiments + plots |
 
## Status
- [x] Baselines: zero-shot, local-only, centralized
- [ ] **Fix eval harness**: broken ATT&CK metric, triage leakage + macro-F1
- [ ] **Run the 12 federated experiments** (core contribution — not yet run)
- [ ] Multi-seed + DP privacy-accounting check
- [ ] Test & document the HuggingFace model

## Working setup
- **Code + results** → this repo.
- **GPU execution** → Colab: *File → Open notebook → GitHub tab*, run, then
  *File → Save a copy in GitHub* to commit back.
- **Big data** (curated corpus, checkpoints) → Google Drive / HF, gitignored here.
 