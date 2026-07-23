"""Analysis — turn results/<id>.json into the paper's table and figures.

CPU only, no model loading. Re-run as often as you like; it never touches a GPU.
Reads whatever results exist and skips figures it can't build yet.

Headline metric per model: the judge's correctness if available, else the
verdict Macro-F1, else mean ROUGE-L — so plots work offline (no judge) too.
"""
from __future__ import annotations

import glob
import json
import math
import os

from .config import Config


# --------------------------------------------------------------------------- #
# loading + metric selection (pure; unit-tested)
# --------------------------------------------------------------------------- #
def load_results(cfg: Config) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(cfg.results_dir, "*.json")))]


def headline_metric(tasks: dict) -> float:
    """One number summarising a model across tasks (see module docstring)."""
    v = tasks.get("verdict", {})
    if "judge_correct" in v:
        return float(v["judge_correct"])
    if "verdict_macro_f1" in v:
        return float(v["verdict_macro_f1"])
    rouges = [m["rouge_l"] for m in tasks.values() if "rouge_l" in m]
    return float(sum(rouges) / len(rouges)) if rouges else 0.0


def _epsilon(r: dict) -> float:
    e = r.get("epsilon", "inf")
    return float("inf") if e in ("inf", None) else float(e)


# --------------------------------------------------------------------------- #
# comparison table
# --------------------------------------------------------------------------- #
def comparison_table(cfg: Config, results: list[dict]):
    import pandas as pd
    rows = []
    for r in results:
        tasks = r.get("tasks", {})
        row = {"id": r.get("id"), "stage": r.get("stage"), "headline": round(headline_metric(tasks), 3)}
        for t, m in tasks.items():
            if "verdict_macro_f1" in m:
                row[f"{t}_f1"] = round(m["verdict_macro_f1"], 3)
            if "judge_correct" in m:
                row[f"{t}_judge"] = round(m["judge_correct"], 3)
            if "rouge_l" in m:
                row[f"{t}_rouge"] = round(m["rouge_l"], 3)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("headline", ascending=False)
    out = os.path.join(cfg.figures_dir, "comparison_table.csv")
    df.to_csv(out, index=False)
    print("wrote", out)
    return df


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _ax():
    import matplotlib
    matplotlib.use("Agg")           # headless
    import matplotlib.pyplot as plt
    return plt


def plot_ablation(cfg: Config, results: list[dict]):
    """A/B/C(/D): headline metric by DAPT source. The C-B gap is the story."""
    rows = {r["row"]: headline_metric(r["tasks"]) for r in results if r.get("stage") == "task"}
    if not rows:
        print("  ablation: no task rows yet"); return
    plt = _ax()
    order = [k for k in ["A", "B", "C", "D"] if k in rows]
    labels = {"A": "A: no DAPT", "B": "B: local DAPT", "C": "C: federated DAPT", "D": "D: centralized"}
    plt.figure(figsize=(7, 5))
    plt.bar([labels[k] for k in order], [rows[k] for k in order], color="#534AB7")
    plt.ylabel("headline metric"); plt.title("Ablation: value of the DAPT source")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "ablation.png"); plt.savefig(p, dpi=150); plt.close()
    print("wrote", p)
    if "C" in rows and "B" in rows:
        print(f"  headline: C-B (value of federating) = {rows['C'] - rows['B']:+.3f}")


def plot_privacy_utility(cfg: Config, results: list[dict]):
    """Utility vs privacy budget epsilon for the FedAvg DP sweep."""
    dp = [r for r in results if str(r.get("id", "")).startswith("dapt_fedavg_eps_")]
    if len(dp) < 2:
        print("  privacy-utility: need >=2 eps runs"); return
    pts = sorted((_epsilon(r), headline_metric(r["tasks"])) for r in dp)
    plt = _ax(); plt.figure(figsize=(7, 5))
    xs, ys = zip(*pts)
    plt.plot(xs, ys, "o-", color="#534AB7", label="FedDAPT + DP")
    ref = next((r for r in results if r.get("id") == "dapt_fedavg_no_dp"), None)
    if ref:
        plt.axhline(headline_metric(ref["tasks"]), ls="--", color="#1D9E75", label="no DP (eps=inf)")
    plt.xlabel("privacy budget epsilon (lower = more private)")
    plt.ylabel("headline metric"); plt.title("Privacy-utility tradeoff")
    plt.legend(); plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "privacy_utility.png"); plt.savefig(p, dpi=150); plt.close()
    print("wrote", p)


def plot_byzantine(cfg: Config, results: list[dict]):
    """Grouped bars: under each attack, headline metric per aggregator.
    Expect FedAvg to drop and Krum/trimmed-mean to hold."""
    byz = [r for r in results if r.get("attack", "none") not in ("none", None)]
    if not byz:
        print("  byzantine: no attack runs yet"); return
    attacks = sorted({r["attack"] for r in byz})
    aggs = sorted({r.get("aggregator", "?") for r in byz})
    plt = _ax(); import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(aggs), 1)
    for j, agg in enumerate(aggs):
        vals = []
        for atk in attacks:
            m = next((r for r in byz if r["attack"] == atk and r.get("aggregator") == agg), None)
            vals.append(headline_metric(m["tasks"]) if m else 0.0)
        ax.bar(np.arange(len(attacks)) + j * width, vals, width, label=agg)
    ax.set_xticks(np.arange(len(attacks)) + width * (len(aggs) - 1) / 2)
    ax.set_xticklabels(attacks); ax.set_ylabel("headline metric")
    ax.set_title("Byzantine robustness: aggregator vs attack"); ax.legend()
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "byzantine.png"); plt.savefig(p, dpi=150); plt.close()
    print("wrote", p)


def plot_learning_curves(cfg: Config, results: list[dict]):
    """Per-round validation perplexity for DAPT runs (convergence evidence)."""
    curves = [(r["id"], r["round_log"]) for r in results
              if r.get("stage") == "dapt" and r.get("round_log")]
    if not curves:
        print("  learning curves: no round logs yet"); return
    plt = _ax(); plt.figure(figsize=(9, 5))
    for rid, log in curves[:10]:
        xs = [d["round"] for d in log]
        ys = [d.get("val_ppl") for d in log]
        if any(y is not None and not math.isnan(y) for y in ys):
            plt.plot(xs, ys, "o-", label=rid, linewidth=1.2)
    plt.xlabel("round"); plt.ylabel("validation perplexity")
    plt.title("DAPT convergence"); plt.legend(fontsize=7); plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "learning_curves.png"); plt.savefig(p, dpi=150); plt.close()
    print("wrote", p)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_analysis(cfg: Config):
    cfg.ensure_dirs()
    results = load_results(cfg)
    if not results:
        raise SystemExit("no results yet — run scripts/evaluate.py first")
    print(f"loaded {len(results)} result files")
    comparison_table(cfg, results)
    plot_ablation(cfg, results)
    plot_privacy_utility(cfg, results)
    plot_byzantine(cfg, results)
    plot_learning_curves(cfg, results)
    print("analysis done — figures in", cfg.figures_dir)
