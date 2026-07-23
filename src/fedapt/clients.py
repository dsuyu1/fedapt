"""Partition data into N non-IID clients (organisations).

Each org owns two things:
  * a PROSE shard   -> its slice of the DAPT corpus (Stage 1 federation input)
  * a LOG slice     -> its private telemetry (Stage 2 task grounding + eval)

Non-IID is created with a Dirichlet(alpha) skew over a label (subdomain for
prose, log domain for logs) — the standard FL heterogeneity knob. Small alpha =
each label concentrated in a few clients.

LOG data comes from the public `splunk/attack_data` repo (git-lfs, ~9GB). Clone
it separately and point FEDDAPT_ATTACK_DATA at the checkout; this module only
reads it. Benign examples (needed for the "is this bad" task) are a known gap —
attack_data is all attack telemetry; wire a benign source via `benign_dir`.
"""
from __future__ import annotations

import glob
import json
import os
import random

import numpy as np
import yaml

from .config import Config

# map Splunk sourcetypes -> our security domains (for non-IID + task routing)
_SOURCETYPE_DOMAIN = {
    "sysmon": "endpoint", "windows": "endpoint", "wineventlog": "endpoint",
    "security": "endpoint", "powershell": "endpoint", "crowdstrike": "endpoint",
    "zeek": "network", "bro": "network", "firewall": "network", "dns": "network",
    "suricata": "network", "netflow": "network",
    "aws": "cloud", "cloudtrail": "cloud", "azure": "cloud", "gcp": "cloud",
    "o365": "cloud", "okta": "cloud",
}


def _domain_of(sourcetype: str) -> str:
    s = (sourcetype or "").lower()
    for key, dom in _SOURCETYPE_DOMAIN.items():
        if key in s:
            return dom
    return "general"


# --------------------------------------------------------------------------- #
# generic non-IID partition
# --------------------------------------------------------------------------- #
def dirichlet_partition(items, label_fn, n_clients, alpha, seed=42, min_items=0):
    """Split `items` across n clients with Dirichlet(alpha) label skew."""
    rng = np.random.default_rng(seed)
    buckets: dict = {}
    for it in items:
        buckets.setdefault(label_fn(it), []).append(it)
    clients = [[] for _ in range(n_clients)]
    for group in buckets.values():
        group = list(group); rng.shuffle(group)
        if not group:
            continue
        p = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(group)).astype(int)[:-1]
        for i, part in enumerate(np.split(np.array(group, dtype=object), cuts)):
            clients[i].extend(list(part))
    for i in range(n_clients):                       # keep no client starved
        if len(clients[i]) < min_items:
            big = max(range(n_clients), key=lambda j: len(clients[j]))
            need = min_items - len(clients[i])
            clients[i].extend(clients[big][:need]); del clients[big][:need]
    for c in clients:
        rng.shuffle(c)
    return clients


# --------------------------------------------------------------------------- #
# log data (attack_data)
# --------------------------------------------------------------------------- #
def load_attack_data(attack_data_dir: str, max_lines_per_log: int = 40) -> list[dict]:
    """Read attack_data YAML metadata + .log files -> labelled log records.

    Returns [{log, technique, sourcetype, domain, is_malicious}]. All attack_data
    telemetry is malicious; benign records must be added separately.
    """
    if not attack_data_dir or not os.path.isdir(attack_data_dir):
        print("  attack_data not found — set FEDDAPT_ATTACK_DATA to a checkout. "
              "Returning [] (log tasks will be empty).")
        return []
    records = []
    for yml in glob.glob(f"{attack_data_dir}/datasets/**/*.yml", recursive=True):
        try:
            meta = yaml.safe_load(open(yml, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        techniques = meta.get("mitre_technique", []) or ["unknown"]
        for ds in meta.get("datasets", []):
            rel = str(ds.get("path", "")).lstrip("/")
            log_path = os.path.join(attack_data_dir, rel)
            if not os.path.exists(log_path):
                continue
            try:
                lines = [ln.strip() for ln in open(log_path, errors="ignore") if ln.strip()]
            except Exception:
                continue
            if not lines:
                continue
            snippet = "\n".join(lines[:max_lines_per_log])
            records.append({
                "log": snippet,
                "technique": techniques[0],
                "sourcetype": ds.get("sourcetype", ""),
                "domain": _domain_of(ds.get("sourcetype", "")),
                "is_malicious": True,
            })
    print(f"  loaded {len(records)} attack_data log records")
    return records


def load_benign(benign_dir: str, max_lines_per_log: int = 40) -> list[dict]:
    """Read a directory of BENIGN telemetry (same .log/.json format) -> records
    labelled is_malicious=False. This is the negative class for the verdict task.

    Recommended public sources (drop a checkout and point FEDDAPT_BENIGN_DATA at it):
    OTRF Security-Datasets 'environment'/baseline captures, an attack_range benign
    baseline run, or a public benign network dataset for the network domain. If no
    dir is given, tasks.build_tasks can synthesise negatives with the teacher LLM.
    """
    if not benign_dir or not os.path.isdir(benign_dir):
        print("  no benign dir (FEDDAPT_BENIGN_DATA) — verdict task needs negatives "
              "(provide a dir, or a teacher to synthesise them).")
        return []
    records = []
    for lp in (glob.glob(f"{benign_dir}/**/*.log", recursive=True) +
               glob.glob(f"{benign_dir}/**/*.json", recursive=True)):
        try:
            lines = [ln.strip() for ln in open(lp, errors="ignore") if ln.strip()]
        except Exception:
            continue
        if not lines:
            continue
        name = os.path.basename(lp)
        records.append({"log": "\n".join(lines[:max_lines_per_log]), "technique": "benign",
                        "sourcetype": name, "domain": _domain_of(name), "is_malicious": False})
    print(f"  loaded {len(records)} benign log records")
    return records


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_clients(cfg: Config) -> dict:
    """Shard prose into DAPT clients and logs into task clients. Writes JSON."""
    cfg.ensure_dirs()
    out = {"prose": [], "logs": [], "benign": []}

    # 1) prose shards for federated DAPT
    corpus_path = os.path.join(cfg.corpus_dir, "prose_corpus.jsonl")
    prose = [json.loads(l) for l in open(corpus_path, encoding="utf-8")] if os.path.exists(corpus_path) else []
    if prose:
        shards = dirichlet_partition(
            prose, lambda d: d.get("subdomain", "general"),
            cfg.n_clients, cfg.dirichlet_alpha, cfg.seed, cfg.min_docs_per_client)
        for i, shard in enumerate(shards):
            p = os.path.join(cfg.clients_dir, f"client_{i}_prose.json")
            json.dump([d["text"] for d in shard], open(p, "w"))
            out["prose"].append((f"client_{i}", len(shard)))
        print("prose shards:", out["prose"])
    else:
        print("  no prose corpus yet — run corpus.build_corpus first.")

    # 2) log slices for per-org task grounding
    logs = load_attack_data(cfg.attack_data_dir)
    if logs:
        slices = dirichlet_partition(logs, lambda r: r["domain"],
                                     cfg.n_clients, cfg.dirichlet_alpha, cfg.seed)
        for i, sl in enumerate(slices):
            p = os.path.join(cfg.clients_dir, f"client_{i}_logs.json")
            json.dump(sl, open(p, "w"))
            out["logs"].append((f"client_{i}", len(sl)))
        print("log slices:", out["logs"])

    # 3) benign log slices (negative class for the verdict task)
    benign = load_benign(cfg.benign_data_dir)
    if benign:
        slices = dirichlet_partition(benign, lambda r: r["domain"],
                                     cfg.n_clients, cfg.dirichlet_alpha, cfg.seed)
        for i, sl in enumerate(slices):
            json.dump(sl, open(os.path.join(cfg.clients_dir, f"client_{i}_benign.json"), "w"))
            out["benign"].append((f"client_{i}", len(sl)))
        print("benign slices:", out["benign"])
    return out
