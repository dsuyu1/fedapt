"""Harvest vendor threat-intel write-ups from official RSS feeds into the DAPT
prose corpus input dir. Run before build_data.

    python scripts/fetch_vendor_rss.py                 # feed summaries -> FEDDAPT_VENDOR_DATA
    python scripts/fetch_vendor_rss.py --full          # also fetch full article text (robots-respecting)
    python scripts/fetch_vendor_rss.py --out ./vendor --limit 30
    python scripts/fetch_vendor_rss.py --feeds my_feeds.json   # override feed list

Output: one {title, content, vendor} JSON per article, read automatically by
corpus.collect_vendor_writeups when FEDDAPT_VENDOR_DATA points at --out.
Incremental — re-run any time to pick up new posts. Respect each source's terms.
Needs `pip install -e ".[fetch]"` for robust parsing/extraction.
"""
import argparse
import os

from fedapt.config import load_config
from fedapt import vendor_feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="output dir (default: FEDDAPT_VENDOR_DATA)")
    ap.add_argument("--feeds", default=os.environ.get("FEDDAPT_VENDOR_FEEDS", ""),
                    help="JSON file of {vendor: feed_url} to override the defaults")
    ap.add_argument("--limit", type=int, default=50, help="max articles per feed")
    ap.add_argument("--full", action="store_true", help="fetch full article text (slower)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args()

    cfg = load_config()
    out = a.out or cfg.vendor_data_dir or os.path.join(cfg.root, "vendor_articles")
    feeds = vendor_feeds.load_feeds(a.feeds or None)
    print(f"out = {out} | {len(feeds)} feeds | full_text={a.full}")
    vendor_feeds.harvest(out, feeds=feeds, limit=a.limit, full_text=a.full, delay=a.delay)
    print(f"\nSet FEDDAPT_VENDOR_DATA={out} then run: python scripts/build_data.py")


if __name__ == "__main__":
    main()
