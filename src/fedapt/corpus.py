"""Build the PROSE DAPT corpus — the shared knowledge substrate.

Natural-language security text only (no logs). Each doc is:
    {"text": str, "source": str, "subdomain": str, "id": str}

Sources are public and prose-shaped so the model learns security *language and
reasoning*, which is what generalises across orgs and is safe to federate.

Add prose sources here (DFIR Report, Atomic Red Team descriptions, CIS/NIST,
security Q&A) as you gather them — one collector function each.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable

import requests
import yaml

from .config import Config


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _doc(text: str, source: str, subdomain: str) -> dict:
    return {"text": text, "source": source, "subdomain": subdomain, "id": _doc_id(text)}


# --------------------------------------------------------------------------- #
# collectors — each returns list[dict]; keep them independent and pure
# --------------------------------------------------------------------------- #
def collect_attack(cache_dir: str) -> list[dict]:
    """MITRE ATT&CK Enterprise (STIX): techniques, procedures, groups, software."""
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    cache = os.path.join(cache_dir, "enterprise-attack.json")
    if os.path.exists(cache):
        data = json.load(open(cache))
    else:
        data = requests.get(url, timeout=120).json()
        json.dump(data, open(cache, "w"))
    docs = []
    for o in data.get("objects", []):
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        desc = (o.get("description") or "").strip()
        if len(desc) < 20:
            continue
        name = o.get("name", "")
        ext = (o.get("external_references") or [{}])[0].get("external_id", "")
        t = o.get("type")
        if t == "attack-pattern":
            docs.append(_doc(f"MITRE ATT&CK Technique {ext} ({name}): {desc}", "mitre_attack", "cti"))
        elif t == "relationship" and o.get("relationship_type") == "uses":
            docs.append(_doc(f"MITRE ATT&CK Procedure: {desc}", "mitre_attack", "cti"))
        elif t == "intrusion-set":
            docs.append(_doc(f"MITRE ATT&CK Threat Group {name}: {desc}", "mitre_attack", "cti"))
        elif t in ("malware", "tool"):
            docs.append(_doc(f"MITRE ATT&CK Software {name}: {desc}", "mitre_attack", "cti"))
    return docs


def collect_sigma(work_dir: str) -> list[dict]:
    """SigmaHQ detection rules -> prose, subdomain-labelled by logsource."""
    import glob
    path = os.path.join(work_dir, "sigma")
    if not os.path.exists(path):
        os.system(f"git clone --depth 1 https://github.com/SigmaHQ/sigma.git {path}")
    docs = []
    for fp in glob.glob(f"{path}/**/*.yml", recursive=True):
        try:
            rule = yaml.safe_load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rule, dict):
            continue
        title, desc = rule.get("title", ""), rule.get("description", "")
        if not (title or desc):
            continue
        ls = str(rule.get("logsource", {})).lower()
        if any(k in ls for k in ("process", "sysmon", "powershell", "windows", "registry")):
            sub = "endpoint"
        elif any(k in ls for k in ("firewall", "network", "dns", "proxy", "zeek")):
            sub = "network"
        elif any(k in ls for k in ("aws", "azure", "gcp", "cloud", "o365")):
            sub = "cloud"
        else:
            sub = "general"
        lvl = rule.get("level", "unknown")
        docs.append(_doc(f"Sigma Detection Rule [{lvl.upper()}]: {title}. {desc}".strip(), "sigma", sub))
    return docs


def collect_nvd(api_key: str = "", max_pages: int = 5) -> list[dict]:
    """NVD CVE descriptions. Needs a key for a usable rate limit (see .env)."""
    base = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers = {"apiKey": api_key} if api_key else {}
    docs, start = [], 0
    for _ in range(max_pages):
        try:
            r = requests.get(base, headers=headers,
                             params={"startIndex": start, "resultsPerPage": 2000}, timeout=120)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  NVD stopped: {e}")
            break
        data = r.json()
        for v in data.get("vulnerabilities", []):
            cve = v.get("cve", {})
            desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
            if not desc or ("reserved" in desc.lower() and len(desc) < 120):
                continue
            docs.append(_doc(f"CVE {cve.get('id','')}: {desc}", "nvd", "vulnerability"))
        start += 2000
        if start >= data.get("totalResults", 0):
            break
    return docs


def collect_vendor_writeups(articles_dir: str) -> list[dict]:
    """Vendor threat-intel / IR write-ups as prose — rich security reasoning that
    boosts the explain/QA tasks and is public (safe to federate).

    Reads a directory you populate yourself:
      *.json  -> {"title", "content", "vendor"}
      *.txt   -> filename is the title, file body is the content
    Point FEDDAPT_VENDOR_DATA at it (nested dirs ok). Long articles are fine —
    the DAPT packer splits them into blocks.

    NOTE (licensing): respect each source's terms/copyright. Prefer official
    RSS/Atom feeds, vendor-provided datasets, or clearly-licensed collections;
    don't scrape sites whose ToS forbid it.
    """
    import glob
    if not articles_dir or not os.path.isdir(articles_dir):
        return []
    docs = []
    for fp in glob.glob(f"{articles_dir}/**/*.json", recursive=True):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        content = (data.get("content") or data.get("text") or "").strip()
        if len(content) < 50:
            continue
        vendor = (data.get("vendor") or "vendor_intel").strip().lower() or "vendor_intel"
        title = data.get("title", "")
        docs.append(_doc(f"{vendor.upper()} Threat Report [{title}]: {content}", vendor, "cti"))
    for fp in glob.glob(f"{articles_dir}/**/*.txt", recursive=True):
        try:
            content = open(fp, encoding="utf-8", errors="ignore").read().strip()
        except Exception:
            continue
        if len(content) < 50:
            continue
        title = os.path.splitext(os.path.basename(fp))[0]
        docs.append(_doc(f"Threat Report [{title}]: {content}", "vendor_intel", "cti"))
    print(f"  vendor write-ups: {len(docs)} docs")
    return docs


def collect_cisa_kev() -> list[dict]:
    """CISA Known Exploited Vulnerabilities catalog."""
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    try:
        data = requests.get(url, timeout=60).json()
    except Exception as e:
        print(f"  CISA KEV failed: {e}")
        return []
    docs = []
    for v in data.get("vulnerabilities", []):
        if not v.get("shortDescription"):
            continue
        docs.append(_doc(
            f"CISA Known Exploited Vulnerability {v.get('cveID','')}: "
            f"{v.get('vulnerabilityName','')}. {v.get('shortDescription','')}",
            "cisa_kev", "vulnerability"))
    return docs


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_corpus(cfg: Config, collectors: list[Callable] | None = None) -> str:
    """Run collectors, dedup, write JSONL to corpus_dir. Returns the path."""
    cfg.ensure_dirs()
    if collectors is None:
        collectors = [
            lambda: collect_attack(cfg.corpus_dir),
            lambda: collect_sigma(cfg.scratch),
            lambda: collect_nvd(cfg.secret("NVD_API_KEY"), max_pages=5),
            collect_cisa_kev,
        ]
        if cfg.vendor_data_dir:                        # opt-in: vendor threat-intel prose
            collectors.append(lambda: collect_vendor_writeups(cfg.vendor_data_dir))
    docs, seen = [], set()
    for fn in collectors:
        got = fn()
        print(f"  {getattr(fn, '__name__', 'collector')}: {len(got)} docs")
        for d in got:
            if d["id"] in seen:
                continue
            seen.add(d["id"]); docs.append(d)
    out = os.path.join(cfg.corpus_dir, "prose_corpus.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"prose corpus: {len(docs)} docs -> {out}")
    return out
