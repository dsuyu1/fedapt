"""Offline tests for the RSS harvester (no network — fetch is monkeypatched)."""
import json
import os
import tempfile

from fedapt import vendor_feeds as V

RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel><title>Vendor</title>
  <item>
    <title>Intrusion via ISO</title>
    <link>https://example.com/a</link>
    <content:encoded><![CDATA[<p>The actor used a <b>malicious ISO</b> and then ran PowerShell to establish persistence across the estate.</p>]]></content:encoded>
    <pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Cloud persistence</title>
    <link href="https://example.com/b"/>
    <summary>Attacker created an IAM role in AWS CloudTrail to maintain access to the environment.</summary>
    <updated>2026-01-02T00:00:00Z</updated>
  </entry>
</feed>"""


def test_clean_html():
    assert V.clean_html("<p>hello <b>world</b></p>") == "hello world"
    assert V.clean_html("a &amp; b") == "a & b"


def test_parse_rss_and_atom():
    rss = V.parse_feed(RSS)
    assert rss and rss[0]["title"] == "Intrusion via ISO"
    assert "malicious ISO" in rss[0]["content"] and rss[0]["link"].endswith("/a")
    atom = V.parse_feed(ATOM)
    assert atom and atom[0]["title"] == "Cloud persistence"
    assert atom[0]["link"].endswith("/b") and "CloudTrail" in atom[0]["content"]


def test_harvest_writes_expected_json():
    orig = V._get                                       # patch out the network
    V._get = lambda url, session, timeout=30: RSS
    try:
        out = tempfile.mkdtemp()
        n = V.harvest(out, feeds={"vendor": "http://feed"}, full_text=False, delay=0)
        assert n == 1
        doc = json.load(open(os.path.join(out, os.listdir(out)[0])))
        assert doc["vendor"] == "vendor"
        assert "malicious ISO" in doc["content"] and "<" not in doc["content"]
        # incremental: re-running writes nothing new
        assert V.harvest(out, feeds={"vendor": "http://feed"}, full_text=False, delay=0) == 0
    finally:
        V._get = orig
