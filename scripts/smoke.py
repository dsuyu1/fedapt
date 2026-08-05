"""Smoke test: run the ENTIRE training/eval path on a tiny config, fast.

    python scripts/smoke.py

Purpose: the torch/model code has never actually executed — this shakes out
integration bugs cheaply (a few minutes on a GPU) before you commit to the full
matrix. It uses the real Mistral-7B + QLoRA path (so real bugs surface) but with
2 rounds, 3 local steps, one experiment, one client, and 5 eval records per task.

Isolation: everything writes under a `smoke` run_name — adapters/smoke,
results/smoke, figures/smoke — so it never collides with or pollutes a real run.
Reuses the data you already built (corpus/clients/tasks/eval are shared).

If this finishes and prints a comparison table, your pipeline works end to end.
Requires: pip install -e ".[train,eval]", and the data layer already built.
"""
import glob
import os

from fedapt.config import load_config
from fedapt import model as M, federated, train_tasks, evaluate, analysis
from fedapt.federated import ExperimentSpec


def main():
    cfg = load_config(run_name="smoke", num_rounds=2, local_steps=3)
    cfg.ensure_dirs()
    print(f"SMOKE run — root={cfg.root}  (artifacts under */smoke)")

    if not glob.glob(os.path.join(cfg.clients_dir, "client_*_prose.json")):
        raise SystemExit("no data — run scripts/build_data.py first")

    # --- Stage 1: one federated DAPT experiment (validates the FL loop) ---
    print("\n[1/4] federated DAPT (2 rounds)")
    federated.run_matrix(cfg, [ExperimentSpec("dapt_fedavg_no_dp")])

    # --- Stage 2: one client, rows A (no DAPT) and C (federated DAPT) ---
    print("\n[2/4] task tuning: rows A + C on one client")
    tok = M.load_tokenizer(cfg)
    client0 = sorted(os.path.basename(p).replace("_prose.json", "")
                     for p in glob.glob(os.path.join(cfg.clients_dir, "client_*_prose.json")))[0]
    train_tasks.train_client(cfg, None, "A", client0, tok)                 # no-DAPT
    train_tasks.train_client(cfg, "dapt_fedavg_no_dp", "C", client0, tok)  # federated DAPT

    # --- Eval: zero-shot + adapters, 5 records/task (no judge) ---
    print("\n[3/4] evaluate (5 records/task, no judge)")
    evaluate.run_eval(cfg, use_judge=False, limit=5)

    # --- Analysis: table + figures ---
    print("\n[4/4] analysis")
    analysis.run_analysis(cfg)

    print(f"\n✅ SMOKE PASSED — the full path runs. See {cfg.figures_dir}")
    print("   (delete the */smoke folders before your real run if you like)")


if __name__ == "__main__":
    main()
