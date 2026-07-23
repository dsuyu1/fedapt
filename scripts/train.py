"""Run the training stages (GPU). Needs `pip install -e ".[train]"`.

    python scripts/train.py --stage1     # federated DAPT matrix (Stage 1)
    python scripts/train.py --local      # local DAPT per org   (ablation B)
    python scripts/train.py --stage2     # task tuning A/B/C     (Stage 2)
    python scripts/train.py --all        # all of the above, in order

Everything auto-skips work whose adapter already exists, so it's resumable.
Build the data first: python scripts/build_data.py
"""
import argparse

from fedapt.config import load_config
from fedapt import federated, train_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="federated DAPT matrix")
    ap.add_argument("--local", action="store_true", help="local DAPT (ablation B)")
    ap.add_argument("--stage2", action="store_true", help="task tuning A/B/C")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    print("PROJECT_ROOT =", cfg.root)
    if a.all or a.local:
        federated.run_local_dapt(cfg)      # B needs local adapters before Stage 2
    if a.all or a.stage1:
        federated.run_matrix(cfg)
    if a.all or a.stage2:
        train_tasks.run_task_tuning(cfg)


if __name__ == "__main__":
    main()
