"""Fast unit tests for the pure-Python data layer (no torch, no network)."""
import numpy as np

from fedapt.clients import dirichlet_partition, _domain_of
from fedapt.splits import stratified_split
from fedapt.aggregators import fedavg, trimmed_mean, krum
from fedapt.eval_metrics import parse_verdict


def test_dirichlet_partition_covers_all_items():
    items = [{"d": "endpoint"}] * 60 + [{"d": "network"}] * 30 + [{"d": "cloud"}] * 10
    clients = dirichlet_partition(items, lambda x: x["d"], n_clients=6, alpha=0.5, min_items=5)
    assert len(clients) == 6
    assert sum(len(c) for c in clients) == len(items)
    assert all(len(c) >= 5 for c in clients)          # no starved client


def test_stratified_split_ratios_and_no_overlap():
    rows = [{"input": f"x{i}", "label": "a" if i % 2 else "b"} for i in range(100)]
    tr, va, te = stratified_split(rows, (0.70, 0.15, 0.15))
    assert len(tr) + len(va) + len(te) == 100
    ids = lambda rs: {r["input"] for r in rs}
    assert not (ids(tr) & ids(va)) and not (ids(va) & ids(te)) and not (ids(tr) & ids(te))


def test_aggregators_shapes():
    w = [[np.ones((2, 2)) * k] for k in (1.0, 2.0, 3.0)]
    assert np.allclose(fedavg(w, [1, 1, 1])[0], np.ones((2, 2)) * 2.0)
    assert trimmed_mean(w, [1, 1, 1])[0].shape == (2, 2)
    assert krum(w, [1, 1, 1])[0].shape == (2, 2)


def test_domain_mapping():
    assert _domain_of("XmlWinEventLog:Microsoft-Windows-Sysmon/Operational") == "endpoint"
    assert _domain_of("aws:cloudtrail") == "cloud"
    assert _domain_of("something_unknown") == "general"


def test_parse_verdict():
    assert parse_verdict("... Assessment: malicious") == "malicious"
    assert parse_verdict("Looks fine. Assessment: benign") == "benign"
    assert parse_verdict("no clear signal") is None
