"""Build a human-labeling sheet to VALIDATE THE JUDGE.

Generates candidate answers from the STUDENT model, pairs each with the teacher
REFERENCE, and writes a CSV with a blank `human` column for you to fill:
    human = 1  -> candidate is factually correct w.r.t. the reference
    human = 0  -> it is not
Label BLIND — decide yourself; do not consult the judge (that's the whole point).

No Anthropic/judge needed here — only the student model (GPU or slow CPU). Score
the filled sheet later with scripts/score_judge.py.

    python scripts/make_judge_audit.py                       # candidates from zero-shot base
    python scripts/make_judge_audit.py --adapter dapt_fedavg_no_dp
    python scripts/make_judge_audit.py --n 50 --mix dapt_fedavg_no_dp   # half base, half adapter
"""
import argparse
import csv
import json
import os
import random

from fedapt.config import load_config
from fedapt import model as M
from fedapt import evaluate


def _sample(cfg, tasks, n, seed=0):
    rows = []
    for t in tasks:
        p = os.path.join(cfg.eval_dir, f"{t}_split.json")
        if os.path.exists(p):
            for r in json.load(open(p))["test"]:
                rows.append({"task": t, "question": r["input"], "reference": r["target"]})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _generate_all(cfg, adapter_id, tokenizer, device, items):
    mdl = M.load_eval_model(cfg, adapter_id)
    out = []
    for it in items:
        prompt = evaluate.PROMPTS.get(it["task"], evaluate.PROMPTS["general_qa"]).format(q=it["question"])
        it = dict(it)
        it["candidate"] = evaluate._generate(mdl, tokenizer, prompt, device)
        it["candidate_from"] = adapter_id or "zeroshot"
        out.append(it)
    del mdl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="total items to label")
    ap.add_argument("--adapter", default="", help="generate candidates from this adapter id (default: zero-shot)")
    ap.add_argument("--mix", default="", help="also generate half from this adapter (varied correctness)")
    ap.add_argument("--tasks", default="explain_log,verdict,general_qa")
    a = ap.parse_args()

    import torch
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = M.load_tokenizer(cfg)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    items = _sample(cfg, tasks, a.n)
    if a.mix:                                    # half base, half adapter -> both right & wrong answers
        half = len(items) // 2
        gen = (_generate_all(cfg, None, tok, device, items[:half])
               + _generate_all(cfg, a.mix, tok, device, items[half:]))
    else:
        gen = _generate_all(cfg, a.adapter or None, tok, device, items)

    out = os.path.join(cfg.eval_dir, "judge_audit.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "candidate_from", "question", "reference", "candidate", "human"])
        for i, it in enumerate(gen):
            w.writerow([i, it["task"], it["candidate_from"], it["question"],
                        it["reference"], it["candidate"], ""])   # <- you fill `human` (1/0)
    print(f"wrote {len(gen)} items -> {out}")
    print("Open it, fill the `human` column (1=correct, 0=incorrect) BLIND, then run scripts/score_judge.py")


if __name__ == "__main__":
    main()
