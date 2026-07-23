"""Differential privacy for federated DAPT (DP-FedAvg)

Privacy model (state this in the paper): **user-level DP-FedAvg with full client
participation**. Each round we clip every client's update to an L2 bound and add
Gaussian noise to the aggregate; that is one Gaussian mechanism over the set of
clients, composed over `num_rounds` rounds and tracked with the RDP accountant.
With all clients present every round there is no subsampling amplification, so
`sample_rate = 1.0`. This is a decentralized, round-level guarantee — NOT the
per-example DP-SGD guarantee inside a client.

We use Opacus's accountant. Given a target ε it returns the noise multiplier σ;
given σ it reports the ε actually spent. If Opacus isn't installed we fall back
to the crude map in Config so the pipeline still runs offline (clearly not a
certified ε — a warning is printed).
"""
from __future__ import annotations

import math
import warnings

import numpy as np


def noise_multiplier_for(epsilon, delta, steps, sample_rate=1.0, fallback_map=None) -> float:
    """Noise multiplier σ needed to hit target ε over `steps` rounds (RDP)."""
    if not math.isfinite(epsilon):
        return 0.0
    try:
        from opacus.accountants.utils import get_noise_multiplier
        return float(get_noise_multiplier(
            target_epsilon=float(epsilon), target_delta=float(delta),
            sample_rate=float(sample_rate), steps=int(steps), accountant="rdp"))
    except Exception:
        if fallback_map is not None:
            warnings.warn("Opacus not available — using PLACEHOLDER noise map; "
                          "reported ε is NOT certified. pip install opacus.")
            return float(fallback_map.get(int(epsilon), 1.0))
        raise


def achieved_epsilon(noise_multiplier, delta, steps, sample_rate=1.0) -> float:
    """The ε actually spent by `steps` rounds at this σ — for honest reporting."""
    if noise_multiplier <= 0:
        return float("inf")
    from opacus.accountants import RDPAccountant
    acct = RDPAccountant()
    for _ in range(int(steps)):
        acct.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
    return float(acct.get_epsilon(delta=float(delta)))


def privatize(delta_arrays, clip_norm, noise_multiplier, seed=None):
    """Clip a client update to L2 `clip_norm`, then add Gaussian noise σ·C."""
    rng = np.random.default_rng(seed)
    flat = np.concatenate([d.reshape(-1) for d in delta_arrays])
    scale = min(1.0, clip_norm / (np.linalg.norm(flat) + 1e-12))
    return [d * scale + rng.normal(0, noise_multiplier * clip_norm, size=d.shape)
            for d in delta_arrays]
