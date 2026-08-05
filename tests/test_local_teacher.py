"""The local (Ollama) teacher backend routes correctly and needs no API key."""
import json

import pytest

from fedapt import judge


class _FakeResp:
    def __init__(self, payload):
        self._p = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._p

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_routing_no_key(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"message": {"role": "assistant", "content": "hello world"}})

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # ollama:/local: prefixes must NOT require ANTHROPIC/OPENAI keys
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    llm = judge.make_llm("ollama:gemma2:9b", temperature=0.0)
    out = llm("say hi")

    assert out == "hello world"
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "gemma2:9b"          # prefix stripped, case kept
    assert captured["body"]["messages"][0]["content"] == "say hi"
    assert captured["body"]["options"]["temperature"] == 0.0


def test_ollama_host_override(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        return _FakeResp({"message": {"content": "ok"}})

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("FEDDAPT_OLLAMA_HOST", "http://box:9999")

    judge.make_llm("local:qwen2.5:7b")("hi")
    assert captured["url"] == "http://box:9999/api/chat"


def test_ollama_empty_response_raises(monkeypatch):
    def fake_urlopen(req, timeout=0):
        return _FakeResp({"message": {"content": "   "}})    # blank -> error, not silent

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(Exception):
        judge.make_llm("ollama:gemma2")("x")
