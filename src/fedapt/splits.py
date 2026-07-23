"""Train / validation / test splits + held-out ids for DAPT.

Discipline (see DESIGN.md §8): 70/15/15, stratified where a label exists. Pick
configs by validation, report test once. `heldout_ids` (val + test + lm_val)
are excluded from all training so there is no leakage.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

from .config import Config


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def stratified_split(rows, ratios=(0.70, 0.15, 0.15), label_key="label", seed=42):
    """Return (train, val, test). Stratifies by `label_key` when present."""
    rnd = random.Random(seed)
    buckets: dict = {}
    for r in rows:
        buckets.setdefault(r.get(label_key) or "_", []).append(r)
    tr, va, te = [], [], []
    for items in buckets.values():
        items = list(items); rnd.shuffle(items)
        n = len(items); n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
        tr += items[:n_tr]; va += items[n_tr:n_tr + n_va]; te += items[n_tr + n_va:]
    rnd.shuffle(tr); rnd.shuffle(va); rnd.shuffle(te)
    return tr, va, te


def build_splits(cfg: Config) -> dict:
    """Split every task dataset, build the LM val set, save splits + heldout ids."""
    cfg.ensure_dirs()
    import glob
    heldout: set = set()
    manifest = {}

    for path in glob.glob(os.path.join(cfg.tasks_dir, "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        rows = json.load(open(path))
        for r in rows:
            r["key"] = _key(r["input"])
        tr, va, te = stratified_split(rows, cfg.split_ratios, seed=cfg.seed)
        json.dump({"train": tr, "val": va, "test": te},
                  open(os.path.join(cfg.eval_dir, f"{name}_split.json"), "w"))
        heldout |= {r["key"] for r in va + te}
        manifest[name] = {"train": len(tr), "val": len(va), "test": len(te)}

    # LM validation: held-out raw prose for DAPT perplexity/convergence
    corpus_path = os.path.join(cfg.corpus_dir, "prose_corpus.jsonl")
    if os.path.exists(corpus_path):
        prose = [json.loads(l)["text"] for l in open(corpus_path, encoding="utf-8")]
        random.Random(cfg.seed + 1).shuffle(prose)
        lm_val = prose[:cfg.lm_val_size]
        json.dump(lm_val, open(os.path.join(cfg.eval_dir, "lm_val.json"), "w"))
        heldout |= {_key(t) for t in lm_val}
        manifest["lm_val"] = len(lm_val)

    json.dump(sorted(heldout), open(os.path.join(cfg.eval_dir, "heldout_ids.json"), "w"))
    manifest["heldout_ids"] = len(heldout)
    print("splits:", json.dumps(manifest, indent=2))
    return manifest
