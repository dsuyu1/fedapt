"""Federated aggregation rules. Each takes a list of clients' parameter arrays
(list[np.ndarray], one entry per LoRA tensor) plus client sizes, and returns the
aggregated parameter arrays.

- fedavg        : size-weighted mean (no robustness)
- trimmed_mean  : drop the k highest/lowest per coordinate (soft Byzantine defence)
- krum          : pick the update closest to its neighbours (hard Byzantine defence)
"""
from __future__ import annotations

import numpy as np


def fedavg(weights, sizes):
    total = sum(sizes)
    w = [s / total for s in sizes]
    return [sum(w[j] * weights[j][i] for j in range(len(w)))
            for i in range(len(weights[0]))]


def trimmed_mean(weights, sizes, beta=0.2):
    n = len(weights); k = int(beta * n)
    out = []
    for i in range(len(weights[0])):
        stacked = np.sort(np.stack([weights[j][i] for j in range(n)]), axis=0)
        core = stacked[k:n - k] if n - 2 * k > 0 else stacked
        out.append(core.mean(axis=0))
    return out


def krum(weights, sizes, f=1):
    n = len(weights)
    flat = [np.concatenate([p.reshape(-1) for p in cw]) for cw in weights]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = np.linalg.norm(flat[i] - flat[j])
    m = max(1, n - f - 2)
    scores = [np.sort(dist[i])[1:1 + m].sum() for i in range(n)]
    return weights[int(np.argmin(scores))]


AGGREGATORS = {"fedavg": fedavg, "trimmed_mean": trimmed_mean, "krum": krum}
