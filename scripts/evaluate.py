"""Evaluate everything on the TEST splits and write results/<id>.json.

    python scripts/evaluate.py            # ROUGE/BERTScore + verdict Macro-F1
    python scripts/evaluate.py --judge    # + CLEV LLM-as-judge (primary metric)

The judge needs an independent model and a key:
    ANTHROPIC_API_KEY=...           (or OPENAI_API_KEY)
    FEDDAPT_JUDGE_MODELS=claude-3-5-haiku-20241022,claude-3-5-sonnet-20241022   (two judges -> CLEV voting)
Do NOT use the base model under test as a judge.
"""
import argparse

from fedapt.config import load_config
from fedapt import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="use the CLEV LLM judge")
    a = ap.parse_args()
    evaluate.run_eval(load_config(), use_judge=a.judge)


if __name__ == "__main__":
    main()
