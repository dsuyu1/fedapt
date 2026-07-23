"""Tests for the benign class balancing and the DP accountant fallback."""
from fedapt.tasks import build_verdict
from fedapt.dp import noise_multiplier_for, privatize
import numpy as np


def _log(i, mal):
    return {"log": f"line {i}", "domain": "endpoint", "is_malicious": mal}


def test_verdict_balances_classes():
    mal = [_log(i, True) for i in range(10)]
    ben = [_log(i, False) for i in range(3)]
    out = build_verdict(mal, teacher=None, benign=ben, balance=True)
    labels = [r["label"] for r in out]
    assert labels.count("malicious") == labels.count("benign") == 3   # downsampled to min


def test_verdict_without_benign_is_single_class():
    out = build_verdict([_log(i, True) for i in range(5)], teacher=None, benign=None)
    assert {r["label"] for r in out} == {"malicious"}


def test_dp_noise_multiplier_fallback_and_inf():
    # inf epsilon => no noise
    assert noise_multiplier_for(float("inf"), 1e-5, 20) == 0.0
    # if opacus is present a real sigma comes back; otherwise the fallback map is used.
    sigma = noise_multiplier_for(3, 1e-5, 20, sample_rate=1.0, fallback_map={3: 0.8})
    assert sigma > 0


def test_dp_privatize_clips_large_update():
    delta = [np.ones((4, 4)) * 100.0]                 # huge update
    out = privatize(delta, clip_norm=0.5, noise_multiplier=0.0, seed=0)
    assert np.linalg.norm(out[0]) <= 0.5 + 1e-6       # clipped to the bound
