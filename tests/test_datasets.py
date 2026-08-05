"""Tests for the pluggable dataset-conversion layer (fedapt.datasets)."""
import json
import os

import pytest

from fedapt import datasets as D
from fedapt.config import Config
from fedapt.clients import build_clients


# --------------------------------------------------------------------------- #
# AIT converter
# --------------------------------------------------------------------------- #
def _make_ait(tmp_path, attack_lines):
    """Build a minimal AIT dataset dir: gather/host/logs/auth.log (40 lines) +
    a parallel label file marking `attack_lines` as attacks."""
    logs = tmp_path / "gather" / "h1" / "logs"
    labs = tmp_path / "labels" / "h1" / "logs"
    logs.mkdir(parents=True); labs.mkdir(parents=True)
    (logs / "auth.log").write_text(
        "\n".join(f"event line {i}" for i in range(1, 41)), encoding="utf-8")
    (labs / "auth.log").write_text(
        "\n".join(json.dumps({"line": n, "labels": ["T1110"]}) for n in attack_lines),
        encoding="utf-8")
    return str(tmp_path)


def test_ait_windows_and_labels(tmp_path):
    src = _make_ait(tmp_path, attack_lines=[5])            # only window 1 has an attack
    recs = D.convert_ait(src, window=20)
    assert len(recs) == 2
    mal = [r for r in recs if r["is_malicious"]]
    ben = [r for r in recs if not r["is_malicious"]]
    assert len(mal) == 1 and len(ben) == 1
    assert mal[0]["technique"] == "T1110"
    assert all(r["domain"] == "endpoint" for r in recs)     # auth.log -> endpoint
    assert all(r["sourcetype"] == "auth.log" for r in recs)


def test_ait_missing_gather_fails_loudly(tmp_path):
    with pytest.raises(SystemExit):
        D.convert_ait(str(tmp_path))                        # no gather/ -> loud


# --------------------------------------------------------------------------- #
# CIC converter
# --------------------------------------------------------------------------- #
def test_cic_detects_label_and_classes(tmp_path):
    csv = tmp_path / "flows.csv"
    csv.write_text(
        "Flow Duration,Total Fwd Packets,Label\n"
        "12,3,BENIGN\n"
        "999,80,DDoS\n"
        "5,1,BENIGN\n", encoding="utf-8")
    recs = D.convert_cic(str(csv))
    assert len(recs) == 3
    assert all(r["domain"] == "network" for r in recs)
    mal = [r for r in recs if r["is_malicious"]]
    assert len(mal) == 1 and mal[0]["technique"] == "DDoS"
    assert "Flow Duration=" in recs[0]["log"]               # k=v rendering, label excluded
    assert "Label=" not in recs[0]["log"]


def test_cic_no_label_column_fails(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        D.convert_cic(str(csv))


# --------------------------------------------------------------------------- #
# CloudTrail converter
# --------------------------------------------------------------------------- #
def test_cloudtrail_records_wrapper(tmp_path):
    f = tmp_path / "ct.json"
    f.write_text(json.dumps({"Records": [
        {"eventName": "RunInstances", "eventSource": "ec2.amazonaws.com",
         "userIdentity": {"type": "IAMUser", "arn": "arn:aws:iam::1:user/a"}},
        {"eventName": "GetCallerIdentity", "eventSource": "sts.amazonaws.com"},
    ]}), encoding="utf-8")
    recs = D.convert_cloudtrail(str(f), label="attack")
    assert len(recs) == 2
    assert all(r["domain"] == "cloud" and r["is_malicious"] for r in recs)
    assert recs[0]["technique"] in ("RunInstances", "GetCallerIdentity")


def test_cloudtrail_bad_label(tmp_path):
    f = tmp_path / "ct.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        D.convert_cloudtrail(str(f), label="nonsense")


# --------------------------------------------------------------------------- #
# normalized loader
# --------------------------------------------------------------------------- #
def test_normalized_roundtrip_and_skip(tmp_path):
    recs = [D.make_record("a log line", "endpoint", True, "T1059", "sysmon"),
            D.make_record("normal line", "network", False, "benign", "zeek")]
    out = str(tmp_path / "n.jsonl")
    D.write_jsonl(recs, out)
    with open(out, "a", encoding="utf-8") as f:
        f.write("not json\n")                              # a malformed line
    loaded = D.load_normalized(out)
    assert len(loaded) == 2                                # malformed line skipped
    assert {r["is_malicious"] for r in loaded} == {True, False}


def test_normalized_wrong_schema_raises(tmp_path):
    p = tmp_path / "wrong.jsonl"
    p.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        D.load_normalized(str(p))


# --------------------------------------------------------------------------- #
# build_clients integration: a normalized source supplies BOTH classes
# --------------------------------------------------------------------------- #
def test_build_clients_merges_normalized_source(tmp_path):
    recs = ([D.make_record(f"mal {i}", "endpoint", True, "T1059", "sysmon") for i in range(15)] +
            [D.make_record(f"ben {i}", "endpoint", False, "benign", "sysmon") for i in range(15)])
    src = str(tmp_path / "src.jsonl")
    D.write_jsonl(recs, src)

    cfg = Config(root=str(tmp_path / "root"), n_clients=3,
                 attack_data_dir="", benign_data_dir="", log_sources=src)
    out = build_clients(cfg)

    total_logs = sum(n for _, n in out["logs"])
    total_benign = sum(n for _, n in out["benign"])
    assert total_logs == 15 and total_benign == 15         # split by is_malicious
    # files exist and are non-degenerate
    assert os.path.exists(os.path.join(cfg.clients_dir, "client_0_logs.json"))
    assert os.path.exists(os.path.join(cfg.clients_dir, "client_0_benign.json"))
