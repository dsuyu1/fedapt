"""Build the four task datasets. Input varies; the target is always prose.

Tasks (see DESIGN.md §6):
  1 explain_example  — walk through a scenario/procedure
  2 explain_log      — interpret a telemetry snippet   (org-private, primary)
  3 general_qa       — open security question
  4 verdict          — "is this bad, and why" in paragraph form  (primary)

Targets that a raw log has no natural label for (tasks 1-3, and task-4 rationale)
are *teacher-synthesized*: pass a `teacher(prompt) -> str` callable (an LLM). If
none is given, a deterministic metadata-based fallback is used so the pipeline
runs offline — but those targets are weak; use a real teacher for real runs.

Each record: {"task", "input", "target", "label", "meta"}.
`label` is the hard ground-truth for task 4 ("malicious"/"benign"); None else.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Optional

from .config import Config

Teacher = Optional[Callable[[str], str]]


_FB = {"n": 0}   # count of targets that fell back due to a teacher error (per build)


def _target(teacher: Teacher, prompt: str, fallback: str) -> str:
    if teacher is None:
        return fallback
    try:
        out = teacher(prompt).strip()
        if out:
            return out
    except Exception:
        pass
    _FB["n"] += 1                    # teacher was set but the call failed
    return fallback


def _verdict_prompt(domain: str, log: str) -> str:
    return (f"The following {domain} activity is mostly normal background activity. "
            f"Does it contain evidence of attack activity? Explain your reasoning, then "
            f"end with 'Assessment: attack' or 'Assessment: benign':\n{log}")


# --------------------------------------------------------------------------- #
# task builders
# --------------------------------------------------------------------------- #
def build_explain_example(prose: list[dict], teacher: Teacher = None, n=500, seed=42) -> list[dict]:
    rows = [d for d in prose if d["text"].startswith("MITRE ATT&CK Procedure")]
    random.Random(seed).shuffle(rows)
    out = []
    for d in rows[:n]:
        ex = d["text"].split(":", 1)[-1].strip()
        prompt = f"Explain what is happening in this security scenario:\n{ex}"
        out.append({"task": "explain_example", "input": ex,
                    "target": _target(teacher, prompt, ex),
                    "label": None, "meta": {"source": d["source"]}})
    return out


def build_explain_log(logs: list[dict], teacher: Teacher = None, seed=42) -> list[dict]:
    out = []
    for r in logs:
        prompt = (f"You are a SOC analyst. Explain what this {r['domain']} telemetry "
                  f"shows and what technique it maps to:\n{r['log']}")
        fallback = f"This {r['domain']} telemetry is consistent with MITRE technique {r['technique']}."
        out.append({"task": "explain_log", "input": r["log"],
                    "target": _target(teacher, prompt, fallback),
                    "label": None, "client": r.get("client"),
                    "meta": {"technique": r["technique"], "domain": r["domain"]}})
    return out


def build_general_qa(prose: list[dict], teacher: Teacher = None, n=500, seed=1) -> list[dict]:
    rows = [d for d in prose if d["text"].startswith("MITRE ATT&CK Technique")]
    random.Random(seed).shuffle(rows)
    out = []
    for d in rows[:n]:
        body = d["text"].split(":", 1)[-1].strip()
        name = d["text"].split("(", 1)[-1].split(")", 1)[0] if "(" in d["text"] else "this technique"
        q = f"What is {name} and how do attackers use it?"
        out.append({"task": "general_qa", "input": q,
                    "target": _target(teacher, q, body),
                    "label": None, "meta": {"source": d["source"]}})
    return out


def synthesize_benign(teacher: Teacher, domains, n_per_domain=50) -> list[dict]:
    """SYNTHETIC negatives (disclosed): teacher writes benign telemetry per domain.
    Used only when no real benign dir is available, to give the verdict task a
    negative class. Mark them so they can be reported/excluded separately."""
    if teacher is None:
        return []
    out = []
    for dom in domains:
        for _ in range(n_per_domain):
            prompt = (f"Write a short, realistic snippet of BENIGN {dom} security "
                      f"telemetry showing normal activity (no attack). Output only the log lines.")
            try:
                log = teacher(prompt).strip()
            except Exception:
                continue
            if log:
                out.append({"log": log, "technique": "benign", "domain": dom,
                            "is_malicious": False, "synthetic": True})
    print(f"  synthesised {len(out)} benign negatives (teacher)")
    return out


def build_verdict(logs: list[dict], teacher: Teacher = None, benign: list[dict] | None = None,
                  balance=True, seed=42) -> list[dict]:
    """Paragraph verdict + hard label. attack_data logs = malicious; `benign`
    records (is_malicious=False) are the negative class. `balance` downsamples the
    larger class so Macro-F1 isn't degenerate."""
    mal = list(logs)
    ben = list(benign or [])
    if balance and ben:
        k = min(len(mal), len(ben))
        rnd = random.Random(seed)
        mal, ben = rnd.sample(mal, k), rnd.sample(ben, k)
    elif not ben:
        print("  WARNING: verdict task has no benign class — Macro-F1 will be degenerate. "
              "Set FEDDAPT_BENIGN_DATA or pass a teacher.")
    pool = mal + ben
    random.Random(seed).shuffle(pool)
    out = []
    for r in pool:
        # Record-level label: the technique label is ground truth for the whole
        # record ("an attack occurred somewhere in this activity"), NOT per line.
        # So the question is whether the activity *contains evidence of attack
        # activity*, not whether every line is malicious. (See DESIGN.md note.)
        label = "attack" if r.get("is_malicious", True) else "benign"
        prompt = _verdict_prompt(r.get("domain", ""), r["log"])
        verdict_txt = ("contains evidence of attack activity" if label == "attack"
                       else "shows only normal activity")
        fallback = f"This {r.get('domain','')} activity {verdict_txt}. Assessment: {label}"
        out.append({"task": "verdict", "input": r["log"],
                    "target": _target(teacher, prompt, fallback),
                    "label": label, "client": r.get("client"),
                    "meta": {"domain": r.get("domain", "")}})
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_tasks(cfg: Config, teacher: Teacher = None) -> dict:
    """Assemble all four task datasets from corpus + client logs. Writes JSON."""
    import glob
    cfg.ensure_dirs()
    corpus_path = os.path.join(cfg.corpus_dir, "prose_corpus.jsonl")
    prose = [json.loads(l) for l in open(corpus_path, encoding="utf-8")] if os.path.exists(corpus_path) else []
    logs = []
    for p in sorted(glob.glob(os.path.join(cfg.clients_dir, "client_*_logs.json"))):
        cname = os.path.basename(p).replace("_logs.json", "")   # tag each log with its owner
        for r in json.load(open(p)):
            r["client"] = cname
            logs.append(r)

    # benign negatives for the verdict task: real slices if present, else synthesise
    benign = []
    for p in sorted(glob.glob(os.path.join(cfg.clients_dir, "client_*_benign.json"))):
        cname = os.path.basename(p).replace("_benign.json", "")
        for r in json.load(open(p)):
            r["client"] = cname
            benign.append(r)
    if not benign and teacher is not None:
        domains = sorted({r["domain"] for r in logs}) or ["endpoint", "network", "cloud"]
        benign = synthesize_benign(teacher, domains)

    builders = [
        ("explain_example", lambda: build_explain_example(prose, teacher)),
        ("explain_log", lambda: build_explain_log(logs, teacher)),
        ("general_qa", lambda: build_general_qa(prose, teacher)),
        ("verdict", lambda: build_verdict(logs, teacher, benign=benign)),
    ]
    datasets = {}
    for name, fn in builders:
        _FB["n"] = 0                                   # reset per-task fallback counter
        rows = fn()
        datasets[name] = rows
        json.dump(rows, open(os.path.join(cfg.tasks_dir, f"{name}.json"), "w"))
        note = ""
        if teacher is not None and _FB["n"]:
            note = (f"  ⚠ {_FB['n']}/{len(rows)} targets FELL BACK (teacher errors — "
                    f"likely rate limit; re-run with --resynth {name})")
        print(f"  task {name}: {len(rows)} examples{note}")
    if teacher is None:
        print("  NOTE: no teacher — targets are weak metadata fallbacks. "
              "Pass a teacher (fedapt.judge.make_llm) for real targets.")
    return datasets


def resynthesize_verdict(cfg: Config, teacher: Teacher):
    """Re-generate ONLY the verdict targets in place, reusing the existing inputs,
    labels and splits. Use this when the verdict targets fell back to templates
    (e.g. the teacher was rate-limited on the last, largest task). Cheap: it only
    re-runs the teacher on the verdict records, not the whole pipeline.

    A record keeps its old target if the teacher fails again, so nothing is lost.
    """
    if teacher is None:
        raise SystemExit("resynthesize_verdict needs a teacher")
    split_path = os.path.join(cfg.eval_dir, "verdict_split.json")
    if not os.path.exists(split_path):
        raise SystemExit("no verdict_split.json — run build_data first")
    data = json.load(open(split_path))
    ok = fb = 0
    for part in ("train", "val", "test"):
        for r in data[part]:
            prompt = _verdict_prompt(r.get("meta", {}).get("domain", ""), r["input"])
            _FB["n"] = 0
            r["target"] = _target(teacher, prompt, r["target"])   # keep old on failure
            fb += _FB["n"]; ok += (1 - _FB["n"])
    json.dump(data, open(split_path, "w"))
    tp = os.path.join(cfg.tasks_dir, "verdict.json")             # keep tasks/ in sync
    if os.path.exists(tp):
        json.dump(data["train"] + data["val"] + data["test"], open(tp, "w"))
    print(f"verdict targets re-synthesised: {ok} refreshed, {fb} still fell back -> {split_path}")
