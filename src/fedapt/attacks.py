"""Malicious-client simulation.

Update-level attacks corrupt a client's LoRA delta before aggregation;
data-level poison corrupts the client's training data instead. A malicious
client applies exactly one of these each round.

  sign_flip    : negate + amplify the update (drag training backward)
  scale        : boost the update to dominate the mean (FedAvg's worst case)
  gaussian     : replace the update with random noise
  data_poison  : scramble the client's tokens into garbage (attack via training)

`poison_update` works on numpy delta arrays; `poison_dataset` on a token dataset.
"""
from __future__ import annotations

import numpy as np

UPDATE_ATTACKS = ("sign_flip", "scale", "gaussian")


def poison_update(new_arrays, global_arrays, attack, boost=10.0, seed=0):
    """Return corrupted parameter arrays for a malicious client's update."""
    delta = [na - ga for na, ga in zip(new_arrays, global_arrays)]
    rng = np.random.default_rng(seed)
    if attack == "sign_flip":
        delta = [-boost * d for d in delta]
    elif attack == "scale":
        delta = [boost * d for d in delta]
    elif attack == "gaussian":
        delta = [rng.normal(0, 1.0, size=d.shape).astype(d.dtype) for d in delta]
    else:
        raise ValueError(f"unknown update attack: {attack}")
    return [ga + d for ga, d in zip(global_arrays, delta)]


def poison_dataset(dataset, seed=0):
    """Scramble each row's tokens. `dataset` is an iterable of dicts with
    input_ids / attention_mask; returns an HF Dataset of the same shape."""
    from datasets import Dataset
    import random
    rnd = random.Random(seed)
    rows = []
    for r in dataset:
        ids = list(r["input_ids"]); rnd.shuffle(ids)
        rows.append({"input_ids": ids, "attention_mask": r["attention_mask"], "labels": ids})
    return Dataset.from_list(rows)
