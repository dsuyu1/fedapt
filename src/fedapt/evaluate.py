"""Evaluation driver — one place, run over the TEST splits only.

For each model (zero-shot base, DAPT adapters, task-tuned rows) and each task:
  * generate a prose answer to the test input;
  * ROUGE-L + BERTScore vs the reference   (secondary, surface/embedding overlap);
  * for `verdict`: parse the model's own verdict -> Macro-F1 vs the ground-truth
    benign/malicious label   (a hard number);
  * if a judge is supplied: CLEV LLM-as-judge correctness on the free-form answer
    (primary). Judge model(s) MUST differ from the base model under test.

Task-tuned rows are per client; we evaluate each client's adapter on the SHARED
test set and macro-average -> one result per row. Results land in results/<id>.json
for the analysis notebook.
"""
from __future__ import annotations

import glob
import json
import os

from .config import Config
from . import model as M
from . import eval_metrics as EM

PROMPTS = {  # must match the Stage-2 training format
    "explain_log":     "### Instruction:\n{q}\n\n### Response:\n",
    "verdict":         "### Instruction:\n{q}\n\n### Response:\n",
    "explain_example": "### Instruction:\nExplain what is happening:\n{q}\n\n### Response:\n",
    "general_qa":      "### Instruction:\n{q}\n\n### Response:\n",
}


def _generate(model, tokenizer, prompt, device, max_new_tokens=96) -> str:
    import torch
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


def evaluate_records(model, tokenizer, records, task, device, judges=None) -> dict:
    """Generate + score one task's TEST records. Returns a metrics dict."""
    from sklearn.metrics import f1_score
    preds, refs, judge_items = [], [], []
    y_true, y_pred = [], []
    for r in records:
        prompt = PROMPTS.get(task, PROMPTS["general_qa"]).format(q=r["input"])
        cand = _generate(model, tokenizer, prompt, device)
        preds.append(cand); refs.append(r["target"])
        if task == "verdict":
            y_true.append(r["label"]); y_pred.append(EM.parse_verdict(cand) or "unknown")
        judge_items.append({"question": r["input"], "candidate": cand, "reference": r["target"]})

    metrics = {"rouge_l": EM.rouge_l(preds, refs), "n": len(records)}
    if task == "verdict" and y_true:
        metrics["verdict_macro_f1"] = float(
            f1_score(y_true, y_pred, labels=["malicious", "benign"], average="macro", zero_division=0))
    if judges:
        from .judge import score_free_form
        metrics["judge_correct"] = score_free_form(judges, judge_items)
    return metrics


def _test_records(cfg: Config, task: str, client: str | None = None) -> list[dict]:
    p = os.path.join(cfg.eval_dir, f"{task}_split.json")
    if not os.path.exists(p):
        return []
    rows = json.load(open(p))["test"]
    return rows if client is None else [r for r in rows if r.get("client") == client]


def evaluate_model(cfg, adapter_id, tokenizer, device, judges=None, tasks=None) -> dict:
    """Evaluate one adapter (or zero-shot if None) across all task test sets."""
    model = M.load_eval_model(cfg, adapter_id)
    tasks = tasks or ["explain_log", "verdict", "explain_example", "general_qa"]
    per_task = {}
    for t in tasks:
        recs = _test_records(cfg, t)
        if recs:
            per_task[t] = evaluate_records(model, tokenizer, recs, t, device, judges)
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_task


def _macro_average(dicts: list[dict]) -> dict:
    """Average a list of {task: {metric: val}} dicts across clients."""
    out: dict = {}
    tasks = set().union(*[d.keys() for d in dicts]) if dicts else set()
    for t in tasks:
        vals = [d[t] for d in dicts if t in d]
        keys = set().union(*[v.keys() for v in vals])
        out[t] = {k: float(sum(v.get(k, 0) for v in vals) / len(vals))
                  for k in keys if k != "n"}
    return out


def run_eval(cfg: Config, use_judge: bool = False):
    """Evaluate zero-shot, every DAPT adapter, and each task-tuned row
    (macro-averaged over clients). Writes results/<id>.json."""
    import torch
    cfg.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = M.load_tokenizer(cfg)

    judges = None
    if use_judge:
        from .judge import make_llm
        models = os.environ.get("FEDDAPT_JUDGE_MODELS", cfg.judge_model).split(",")
        judges = [make_llm(m.strip(), cfg.judge_temperature) for m in models if m.strip()]
        print("judges:", models)

    def _write(rid, payload):
        json.dump(payload, open(os.path.join(cfg.results_dir, f"{rid}.json"), "w"), indent=2)
        print("  wrote", rid)

    # 1) zero-shot floor
    if not os.path.exists(os.path.join(cfg.results_dir, "zeroshot.json")):
        print("== zeroshot ==")
        _write("zeroshot", {"id": "zeroshot", "stage": "baseline",
                            "tasks": evaluate_model(cfg, None, tokenizer, device, judges)})

    # 2) DAPT-only adapters (evaluated directly on tasks)
    for mp in sorted(glob.glob(os.path.join(cfg.adapters_dir, "dapt_*", "meta.json"))):
        meta = json.load(open(mp)); rid = meta["id"]
        if os.path.exists(os.path.join(cfg.results_dir, f"{rid}.json")):
            continue
        print("==", rid, "==")
        _write(rid, {**meta, "tasks": evaluate_model(cfg, rid, tokenizer, device, judges)})

    # 3) task-tuned rows: group task_<row>__<client>, macro-average over clients
    rows: dict = {}
    for mp in sorted(glob.glob(os.path.join(cfg.adapters_dir, "task_*", "meta.json"))):
        meta = json.load(open(mp)); rows.setdefault(meta["row"], []).append(meta["id"])
    for row, ids in rows.items():
        rid = f"task_row_{row}"
        if os.path.exists(os.path.join(cfg.results_dir, f"{rid}.json")):
            continue
        print("== task row", row, "==")
        per_client = [evaluate_model(cfg, i, tokenizer, device, judges) for i in ids]
        _write(rid, {"id": rid, "stage": "task", "row": row, "n_clients": len(ids),
                     "tasks": _macro_average(per_client)})
    print("evaluation done — results in", cfg.results_dir)
