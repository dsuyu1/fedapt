"""Harvest vendor threat-intel / IR write-ups from official RSS/Atom feeds.

RSS is the license-respecting path: a feed is what the publisher chooses to
syndicate. We read the feed's own content (or, with --full, fetch the linked
article and extract its main text, honouring robots.txt). Output is one JSON per
article in the `{title, content, vendor}` shape that `corpus.collect_vendor_writeups`
reads — so `scripts/build_data.py` picks them up with no extra wiring.

Dependency-light: `requests` (core) fetches; `feedparser` is used if installed,
else a small stdlib RSS/Atom parser. Full-text extraction uses `trafilatura`
then `bs4` if available, else a naive tag strip. Feed URLs change over time —
override with a JSON file ({vendor: url}) via `--feeds` or FEDDAPT_VENDOR_FEEDS.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import requests

UA = "FeDAPT-research/0.1 (+https://github.com/YOUR_USERNAME/fedapt)"

# Curated official feeds. Verify/adjust — vendors move these occasionally.
DEFAULT_FEEDS = {
    "dfir_report":   "https://thedfirreport.com/feed/",
    "unit42":        "https://unit42.paloaltonetworks.com/feed/",
    "talos":         "https://blog.talosintelligence.com/rss/",
    "red_canary":    "https://redcanary.com/blog/feed/",
    "crowdstrike":   "https://www.crowdstrike.com/blog/feed/",
    "sentinelone":   "https://www.sentinelone.com/labs/feed/",
    "securelist":    "https://securelist.com/feed/",
    "checkpoint":    "https://research.checkpoint.com/feed/",
    "rapid7":        "https://www.rapid7.com/blog/rss/",
    "welivesecurity": "https://www.welivesecurity.com/en/rss/feed/",
    "microsoft":     "https://www.microsoft.com/en-us/security/blog/feed/",
}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def clean_html(s: str) -> str:
    """Strip tags + unescape entities + collapse whitespace."""
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s or "")
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """Return [{title, link, content, published}] from RSS or Atom bytes."""
    try:
        import feedparser
        f = feedparser.parse(xml_bytes)
        out = []
        for e in f.entries:
            content = ""
            if e.get("content"):
                content = e["content"][0].get("value", "")
            content = content or e.get("summary", "") or e.get("description", "")
            out.append({"title": e.get("title", ""), "link": e.get("link", ""),
                        "content": content, "published": e.get("published", "")})
        return out
    except Exception:
        return _parse_stdlib(xml_bytes)


def _parse_stdlib(xml_bytes: bytes) -> list[dict]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes)
    local = lambda e: e.tag.split("}")[-1]
    out = []
    for it in (e for e in root.iter() if local(e) in ("item", "entry")):
        d = {"title": "", "link": "", "content": "", "published": ""}
        for ch in it:
            t = local(ch)
            if t == "title":
                d["title"] = (ch.text or "").strip()
            elif t == "link":
                d["link"] = ch.get("href") or (ch.text or "").strip()
            elif t in ("encoded", "content"):          # content:encoded / atom content
                txt = "".join(ch.itertext()) if list(ch) else (ch.text or "")
                if txt.strip():
                    d["content"] = txt
            elif t in ("description", "summary") and not d["content"]:
                d["content"] = ch.text or ""
            elif t in ("pubDate", "published", "updated") and not d["published"]:
                d["published"] = (ch.text or "").strip()
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def _get(url: str, session: requests.Session, timeout=30) -> bytes | None:
    try:
        r = session.get(url, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"    fetch failed: {url} ({e})")
        return None


def _robots_ok(url: str, session: requests.Session) -> bool:
    try:
        p = urlparse(url)
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:
        return True                                    # no robots -> allowed


def extract_article(url: str, session: requests.Session) -> str:
    """Fetch the article and return main text (robots-respecting)."""
    if not _robots_ok(url, session):
        print(f"    robots.txt disallows {url}")
        return ""
    raw = _get(url, session)
    if not raw:
        return ""
    for extractor in ("trafilatura", "bs4"):
        try:
            if extractor == "trafilatura":
                import trafilatura
                txt = trafilatura.extract(raw.decode("utf-8", "ignore")) or ""
            else:
                from bs4 import BeautifulSoup
                txt = BeautifulSoup(raw, "html.parser").get_text(" ")
            if txt and len(txt) > 200:
                return re.sub(r"\s+", " ", txt).strip()
        except Exception:
            continue
    return clean_html(raw.decode("utf-8", "ignore"))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def harvest(out_dir: str, feeds: dict | None = None, limit=50, full_text=False,
            delay=1.0, session=None) -> int:
    """Pull each feed -> write one {title,content,vendor} JSON per article.
    Incremental: articles already downloaded (by link hash) are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    feeds = feeds or DEFAULT_FEEDS
    session = session or requests.Session()
    written = 0
    for vendor, url in feeds.items():
        print(f"[{vendor}] {url}")
        raw = _get(url, session)
        if not raw:
            continue
        for e in parse_feed(raw)[:limit]:
            link = e.get("link", "")
            key = hashlib.md5((link or e.get("title", "")).encode()).hexdigest()[:12]
            path = os.path.join(out_dir, f"{vendor}_{key}.json")
            if os.path.exists(path):
                continue
            content = clean_html(e.get("content", ""))
            if full_text and link:
                full = extract_article(link, session)
                if len(full) > len(content):
                    content = full
                time.sleep(delay)                      # be polite between article fetches
            if len(content) < 50:
                continue
            json.dump({"title": e.get("title", ""), "content": content, "vendor": vendor,
                       "link": link, "published": e.get("published", "")},
                      open(path, "w", encoding="utf-8"))
            written += 1
        time.sleep(delay)
    print(f"harvested {written} new articles -> {out_dir}")
    return written


def load_feeds(path: str | None) -> dict:
    """Optional {vendor: url} override file (JSON)."""
    if path and os.path.exists(path):
        return json.load(open(path))
    return DEFAULT_FEEDS
