"""Score the CLEV judge against your human labels (run after make_judge_audit.py).

Reads eval/judge_audit.csv (you filled the `human` column with 1/0), runs the
judge on the same items, and reports Cohen's kappa + macro-F1 of judge-vs-human.
Admit the judge only if it clears the CLEV bar (kappa >= 0.6, macro-F1 >= 0.85).

Needs the judge model(s) + key:  FEDDAPT_JUDGE_MODELS, ANTHROPIC_API_KEY (or OPENAI).

    python scripts/score_judge.py
"""
import csv
import os

from fedapt.config import load_config
from fedapt.judge import make_llm, validate_judge


def main():
    cfg = load_config()
    path = os.path.join(cfg.eval_dir, "judge_audit.csv")
    if not os.path.exists(path):
        raise SystemExit("no judge_audit.csv — run scripts/make_judge_audit.py and label it first")

    labeled = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        h = (row.get("human") or "").strip()
        if h in ("0", "1"):                       # skip unlabeled rows
            labeled.append({"question": row["question"], "candidate": row["candidate"],
                            "reference": row["reference"], "human": h == "1"})
    if len(labeled) < 20:
        print(f"WARNING: only {len(labeled)} labeled rows — aim for ~50 for a stable estimate.")
    if not labeled:
        raise SystemExit("no labeled rows (fill the `human` column with 1/0)")

    models = os.environ.get("FEDDAPT_JUDGE_MODELS", cfg.judge_model).split(",")
    judges = [make_llm(m.strip(), cfg.judge_temperature) for m in models if m.strip()]
    print(f"judges: {models} | labeled items: {len(labeled)}")

    res = validate_judge(judges, labeled)
    print(f"\nCohen's kappa : {res['cohen_kappa']:.3f}   (CLEV bar >= 0.60)")
    print(f"macro-F1      : {res['macro_f1']:.3f}   (CLEV bar >= 0.85)")
    ok = res["cohen_kappa"] >= 0.6 and res["macro_f1"] >= 0.85
    print(f"\n{'✅ judge ADMITTED — trust its scores' if ok else '❌ judge below bar — pick a stronger judge model or refine the prompt'}")


if __name__ == "__main__":
    main()
