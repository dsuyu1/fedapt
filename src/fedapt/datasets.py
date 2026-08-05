"""Convert public, paper-backed log datasets into FeDAPT's internal record schema.

This is the pluggable log-source layer that replaces the Splunk Attack Range /
`attack_data`-only path. Any dataset — AIT, CIC-IDS, CloudTrail detonations — is
converted ONCE into a normalized `.jsonl` file, and the pipeline reads those via
`FEDDAPT_LOG_SOURCES`. A single normalized file can carry BOTH classes (each
record has its own `is_malicious`), which is exactly what the verdict task needs.

Internal record schema (one JSON object per line in the .jsonl):
    {
      "log":          str,   # the raw telemetry snippet (what the model reads)
      "technique":    str,   # MITRE technique / attack name, or "benign"
      "sourcetype":   str,   # provenance tag (e.g. "auth.log", "cic-flow")
      "domain":       str,   # "endpoint" | "network" | "cloud" | "general"
      "is_malicious": bool,  # HARD label -> verdict ground truth
    }
This mirrors what `clients.load_attack_data` / `load_benign` already emit, so the
rest of the pipeline is unchanged.

Design rule learned the hard way: **fail loudly on a schema mismatch.** A parser
that silently yields zero (or garbage) records is how you end up training on the
wrong thing. Every converter raises if it produced nothing usable.

Paper anchors (cite these):
  AIT        Landauer et al., "Maintainable Log Datasets for Evaluation of IDS",
             IEEE TDSC 2022.  Zenodo: 10.5281/zenodo.5789064 (V2).
  CIC-IDS    Sharafaldin et al., "Toward Generating a New Intrusion Detection
             Dataset and Intrusion Traffic Characterization", ICISSP 2018.
  UNSW-NB15  Moustafa & Slay, MilCIS 2015.
  CloudTrail Stratus Red Team detonation logs (Datadog), used by recent CloudTrail
             threat-detection papers.
"""
from __future__ import annotations

import csv
import glob
import gzip
import io
import json
import os
import random

REQUIRED_KEYS = ("log", "technique", "sourcetype", "domain", "is_malicious")
VALID_DOMAINS = {"endpoint", "network", "cloud", "general"}


# --------------------------------------------------------------------------- #
# normalized record helpers
# --------------------------------------------------------------------------- #
def make_record(log: str, domain: str, is_malicious: bool,
                technique: str = "", sourcetype: str = "") -> dict:
    return {
        "log": log,
        "technique": technique or ("benign" if not is_malicious else "unknown"),
        "sourcetype": sourcetype,
        "domain": domain if domain in VALID_DOMAINS else "general",
        "is_malicious": bool(is_malicious),
    }


def _valid(rec: object) -> bool:
    return (isinstance(rec, dict)
            and all(k in rec for k in REQUIRED_KEYS)
            and isinstance(rec.get("log"), str) and rec["log"].strip() != ""
            and isinstance(rec.get("is_malicious"), bool))


def load_normalized(path: str) -> list[dict]:
    """Read one normalized .jsonl file -> list of validated records.

    Skips malformed lines with a count; raises if the file yielded NO valid
    records (a silent-empty source is a bug, not an empty dataset)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"normalized source not found: {path}")
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            if _valid(obj):
                recs.append(obj)
            else:
                bad += 1
    if not recs:
        raise ValueError(
            f"{path}: 0 valid records (of which {bad} malformed). Wrong schema? "
            f"Each line must be a JSON object with keys {REQUIRED_KEYS}.")
    if bad:
        print(f"  {os.path.basename(path)}: {len(recs)} records ({bad} skipped)")
    return recs


def write_jsonl(records: list[dict], out_path: str) -> str:
    """Write normalized records to .jsonl and print a class/domain summary."""
    if not records:
        raise ValueError("refusing to write 0 records — the converter produced nothing")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(records)} records -> {out_path}")
    print("  " + summarize(records))
    return out_path


def summarize(records: list[dict]) -> str:
    mal = sum(1 for r in records if r["is_malicious"])
    doms: dict = {}
    for r in records:
        doms[r["domain"]] = doms.get(r["domain"], 0) + 1
    dom_str = ", ".join(f"{k}:{v}" for k, v in sorted(doms.items()))
    return f"class balance: {mal} malicious / {len(records) - mal} benign | domains: {dom_str}"


# --------------------------------------------------------------------------- #
# small IO utilities
# --------------------------------------------------------------------------- #
def _open_text(path: str):
    """Open a possibly-gzipped text file (AIT ships some logs .gz)."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


# =========================================================================== #
# AIT Log Data Set  (Landauer et al., IEEE TDSC 2022)
# =========================================================================== #
# Layout (per dataset, e.g. .../mail.cup.com/):
#     gather/<host>/logs/<logfile>        raw logs, one event per line
#     labels/<host>/logs/<logfile>        parallel labels; each line references an
#                                         ATTACK line in the log by its 1-based
#                                         line number, plus the attack label(s).
# Lines NOT referenced by a label are benign. We window lines into records and
# mark a window malicious iff it contains >=1 labeled attack line — which is
# exactly the verdict task's record-level question ("does this activity CONTAIN
# evidence of an attack?"), not a per-line claim.
_AIT_DOMAIN = [
    # substring in filename -> domain
    ("audit", "endpoint"), ("auth", "endpoint"), ("syslog", "endpoint"),
    ("messages", "endpoint"), ("secure", "endpoint"), ("sysmon", "endpoint"),
    ("suricata", "network"), ("eve", "network"), ("access", "network"),
    ("apache", "network"), ("nginx", "network"), ("error", "network"),
    ("dnsmasq", "network"), ("dns", "network"), ("named", "network"),
    ("openvpn", "network"), ("traffic", "network"),
]


def _ait_domain(filename: str) -> str:
    f = filename.lower()
    for key, dom in _AIT_DOMAIN:
        if key in f:
            return dom
    return "endpoint"          # AIT hosts are servers; default host logs -> endpoint


def _ait_attack_lines(label_path: str) -> dict:
    """Parse an AIT label file -> {line_number: "label1;label2"} for attack lines.

    AIT V2 label lines are JSON objects with at least 'line' and 'labels'. Some
    variants are CSV-ish; we fall back to a leading integer + remainder."""
    attack: dict = {}
    if not os.path.exists(label_path):
        return attack
    with _open_text(label_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n = None
            labels = ""
            try:
                obj = json.loads(ln)
                n = int(obj.get("line"))
                labs = obj.get("labels", [])
                labels = ";".join(labs) if isinstance(labs, list) else str(labs)
            except Exception:
                # fallback: "<line_no> <rest>"  or  "<line_no>,<labels>,..."
                head = ln.replace(",", " ").split()
                if head and head[0].isdigit():
                    n = int(head[0])
                    labels = " ".join(head[1:])[:120]
            if n is not None:
                attack[n] = labels or "attack"
    return attack


def convert_ait(src: str, window: int = 20, max_benign_per_file: int = 200,
                max_malicious_per_file: int = 400, seed: int = 42) -> list[dict]:
    """Convert one AIT dataset directory (containing gather/ and labels/) into
    normalized records. Windows each log file into `window`-line snippets."""
    gather = os.path.join(src, "gather")
    labels_root = os.path.join(src, "labels")
    if not os.path.isdir(gather):
        raise SystemExit(
            f"convert_ait: no 'gather/' under {src}. Point --src at a single AIT "
            f"dataset dir (the one that contains gather/ and labels/).")
    rng = random.Random(seed)
    out: list[dict] = []
    log_files = [p for p in glob.glob(os.path.join(gather, "**", "*"), recursive=True)
                 if os.path.isfile(p)]
    for lp in log_files:
        rel = os.path.relpath(lp, gather)                 # <host>/logs/<file>
        label_path = os.path.join(labels_root, rel)
        attack = _ait_attack_lines(label_path)
        try:
            with _open_text(lp) as f:
                lines = [ln.rstrip("\n") for ln in f]
        except Exception:
            continue
        lines = [ln for ln in lines if ln.strip()]
        if not lines:
            continue
        fname = os.path.basename(lp)
        domain = _ait_domain(fname)
        mal_here, ben_here = [], []
        for start in range(0, len(lines), window):
            chunk = lines[start:start + window]
            # 1-based line numbers this chunk covers
            nums = range(start + 1, start + len(chunk) + 1)
            hits = [attack[n] for n in nums if n in attack]
            snippet = "\n".join(chunk)
            if hits:
                tech = hits[0].split(";")[0] or "attack"
                mal_here.append(make_record(snippet, domain, True, tech, fname))
            else:
                ben_here.append(make_record(snippet, domain, False, "benign", fname))
        rng.shuffle(ben_here); rng.shuffle(mal_here)
        out += mal_here[:max_malicious_per_file]
        out += ben_here[:max_benign_per_file]
    if not out:
        raise SystemExit(
            f"convert_ait: produced 0 records from {src}. Check the gather/labels "
            f"layout (expected gather/<host>/logs/<file>).")
    return out


# =========================================================================== #
# CIC-IDS2017 / UNSW-NB15  (labeled network-flow CSVs)
# =========================================================================== #
_CIC_BENIGN_VALUES = {"benign", "normal", "0", ""}
_CIC_LABEL_NAMES = {"label", "attack_cat", " label"}


def convert_cic(src: str, max_per_class: int = 4000, max_features: int = 25,
                seed: int = 42) -> list[dict]:
    """Convert CIC-IDS / UNSW-NB15 flow CSV(s) into normalized network records.

    The 'log' text is a compact `k=v` rendering of the flow's features (a SOC
    analyst reading a flow record). Label column is auto-detected; a row is benign
    iff its label is BENIGN/Normal/0. Domain is always 'network'."""
    files = ([src] if os.path.isfile(src)
             else sorted(glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)))
    if not files:
        raise SystemExit(f"convert_cic: no .csv found at {src}")
    rng = random.Random(seed)
    buckets = {True: [], False: []}
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            label_col = next((c for c in reader.fieldnames
                              if c.strip().lower() in _CIC_LABEL_NAMES), None)
            if label_col is None:
                raise SystemExit(
                    f"convert_cic: no label column in {os.path.basename(fp)} "
                    f"(looked for {sorted(_CIC_LABEL_NAMES)}). Columns: "
                    f"{reader.fieldnames[:8]}...")
            feat_cols = [c for c in reader.fieldnames if c != label_col][:max_features]
            for row in reader:
                raw = (row.get(label_col) or "").strip()
                is_mal = raw.lower() not in _CIC_BENIGN_VALUES
                text = " ".join(f"{c.strip()}={(row.get(c) or '').strip()}"
                                for c in feat_cols)
                if not text.strip():
                    continue
                tech = raw if is_mal else "benign"
                buckets[is_mal].append(
                    make_record(text, "network", is_mal, tech, "cic-flow"))
    out = []
    for is_mal, recs in buckets.items():
        rng.shuffle(recs)
        out += recs[:max_per_class]
    if not out:
        raise SystemExit(f"convert_cic: produced 0 records from {src}")
    return out


# =========================================================================== #
# CloudTrail detonation logs  (Stratus Red Team, etc.)
# =========================================================================== #
def _iter_cloudtrail_events(path: str):
    """Yield CloudTrail event dicts from a file that is either {"Records":[...]},
    a bare JSON array, or JSONL (one event per line)."""
    txt = open(path, encoding="utf-8", errors="ignore").read().strip()
    if not txt:
        return
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and isinstance(obj.get("Records"), list):
            yield from obj["Records"]; return
        if isinstance(obj, list):
            yield from obj; return
        if isinstance(obj, dict):
            yield obj; return
    except Exception:
        pass
    for line in txt.splitlines():                          # JSONL fallback
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def convert_cloudtrail(src: str, label: str, max_events: int = 4000,
                       seed: int = 42) -> list[dict]:
    """Convert CloudTrail JSON into normalized cloud records.

    A CloudTrail export is one class at a time: pass label='attack' for a Stratus
    detonation export, label='benign' for a normal-account export. Domain=cloud."""
    if label not in ("attack", "benign"):
        raise SystemExit("convert_cloudtrail: --label must be 'attack' or 'benign'")
    is_mal = label == "attack"
    files = ([src] if os.path.isfile(src)
             else sorted(glob.glob(os.path.join(src, "**", "*.json"), recursive=True) +
                         glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True)))
    if not files:
        raise SystemExit(f"convert_cloudtrail: no .json/.jsonl found at {src}")
    rng = random.Random(seed)
    recs = []
    keep = ("eventTime", "eventName", "eventSource", "awsRegion",
            "sourceIPAddress", "errorCode")
    for fp in files:
        for ev in _iter_cloudtrail_events(fp):
            if not isinstance(ev, dict):
                continue
            slim = {k: ev[k] for k in keep if k in ev}
            ui = ev.get("userIdentity")
            if isinstance(ui, dict):
                slim["userIdentity.type"] = ui.get("type")
                slim["userIdentity.arn"] = ui.get("arn")
            if not slim.get("eventName"):
                continue
            tech = ev.get("eventName") if is_mal else "benign"
            recs.append(make_record(json.dumps(slim), "cloud", is_mal, tech, "cloudtrail"))
    if not recs:
        raise SystemExit(
            f"convert_cloudtrail: produced 0 events from {src} — is this CloudTrail JSON?")
    rng.shuffle(recs)
    return recs[:max_events]


CONVERTERS = {
    "ait": convert_ait,
    "cic": convert_cic,
    "cloudtrail": convert_cloudtrail,
}
