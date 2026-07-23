"""Build the whole data layer end to end.

    python scripts/build_data.py                 # metadata-fallback targets (offline)
    python scripts/build_data.py --teacher gpt-4o-mini   # real synthesized targets

Runs: prose corpus -> non-IID clients (prose shards + log slices) -> 4 task
datasets -> train/val/test splits + held-out ids. Everything lands under
FEDDAPT_ROOT (see .env). Safe to re-run; stages are independent.
"""
import argparse

from fedapt.config import load_config
from fedapt import corpus, clients, tasks, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="", help="LLM model id for target synthesis (optional)")
    ap.add_argument("--skip-corpus", action="store_true", help="reuse existing prose corpus")
    args = ap.parse_args()

    cfg = load_config()
    print("PROJECT_ROOT =", cfg.root)

    teacher = None
    if args.teacher:
        from fedapt.judge import make_llm
        teacher = make_llm(args.teacher, temperature=0.0)
        print("teacher:", args.teacher)

    print("\n== 1. prose corpus ==")
    if not args.skip_corpus:
        corpus.build_corpus(cfg)
    print("\n== 2. clients (prose shards + log slices) ==")
    clients.build_clients(cfg)
    print("\n== 3. task datasets ==")
    tasks.build_tasks(cfg, teacher=teacher)
    print("\n== 4. splits ==")
    splits.build_splits(cfg)
    print("\nData layer built under", cfg.root)


if __name__ == "__main__":
    main()
