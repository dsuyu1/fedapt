"""LLM-as-a-judge, our PRIMARY correctness signal for free-form outputs.

Implements CLEV (Badshah et al. 2025, arXiv:2503.08542): reference-aware judging
that returns a binary verdict + rationale, with lightweight voting — two judges
decide, a third breaks ties only on disagreement (~80-95% cheaper than a fixed
3-judge panel).

Integrity rules (DESIGN.md §8), enforced by convention here:
  * the judge model MUST differ from the base model under test (no self-judging);
  * validate the judge against a small human-labeled set (`validate_judge`)
    before trusting it — report Cohen's kappa + macro-F1;
  * judge at temperature 0.

`make_llm` returns a `judge(question, candidate, reference) -> (bool, rationale)`
callable. Plug in any client (OpenAI shown); it is intentionally thin so you can
swap providers. The same LLM callable can serve as the task *teacher* for target
synthesis (fedapt.tasks).
"""
from __future__ import annotations

import os
import re
from typing import Callable

JUDGE_PROMPT = (
    "You are grading a security assistant's answer for factual correctness.\n"
    "Question:\n{q}\n\nReference answer:\n{ref}\n\nCandidate answer:\n{cand}\n\n"
    "Is the candidate factually correct with respect to the reference? "
    "Reply with 'Verdict: correct' or 'Verdict: incorrect', then one sentence of rationale."
)


def _api_key(name: str) -> str:
    """Read a provider key, pulling in `.env` if the environment is bare.

    Only `load_config()` loads `.env`, so a bare `python -c` / script that
    imports this module directly would otherwise see no key at all.
    """
    key = os.environ.get(name, "").strip()
    if not key:
        from .config import _load_dotenv
        _load_dotenv()
        key = os.environ.get(name, "").strip()
    return key


# Claude models from Opus 4.7 / Sonnet 5 on reject `temperature` with a 400.
# They are deterministic enough for judging without it; the older judge-tier
# models (Haiku 4.5, Sonnet 4.6) still take temperature=0.
_NO_TEMPERATURE = ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
                   "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")


def _with_retry(fn, tries=6, base=1.5):
    """Retry a text->text LLM call on transient errors (rate limit / overload /
    timeout) with exponential backoff, so a blip doesn't silently drop to a
    fallback target. Non-transient errors (e.g. auth) raise immediately."""
    import time

    def wrapped(prompt: str) -> str:
        for i in range(tries):
            try:
                return fn(prompt)
            except Exception as e:
                msg = (type(e).__name__ + " " + str(e)).lower()
                transient = any(k in msg for k in (
                    "rate", "overload", "timeout", "connection", "529", "503",
                    "internal", "unavailable"))
                if not transient or i == tries - 1:
                    raise
                time.sleep(min(60.0, base * (2 ** i)))   # 1.5, 3, 6, 12, 24, 48s
    return wrapped


def _ollama_llm(model: str, temperature: float, host: str) -> Callable[[str], str]:
    """text->text callable backed by a LOCAL Ollama server (stdlib only, no key).

    Great as the offline TEACHER for target synthesis: free, private, and — unlike
    a metered API — it can't rate-limit you into silent fallbacks. First call also
    loads the model into memory, so it can be slow; the retry wrapper tolerates the
    initial connection/timeout while the server warms up."""
    import json
    import urllib.request

    url = host.rstrip("/") + "/api/chat"

    def call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 1024},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:   # local gen can be slow
            data = json.loads(r.read().decode("utf-8"))
        text = (data.get("message") or {}).get("content", "")
        if not text.strip():
            raise RuntimeError(f"empty response from ollama model {model!r}")
        return text
    return _with_retry(call)


# prefixes that route a model name to the local Ollama backend
_LOCAL_PREFIXES = ("ollama:", "ollama/", "local:", "local/")


def make_llm(model: str, temperature: float = 0.0) -> Callable[[str], str]:
    """Return a text->text callable backed by an LLM, wrapped with retry/backoff.

    Backends, selected by the model string:
      * ``ollama:<name>`` / ``local:<name>``  -> local Ollama server (no API key)
      * anything with ``claude`` (or an ANTHROPIC_API_KEY present) -> Anthropic
      * otherwise -> OpenAI
    e.g. ``make_llm("ollama:gemma2:9b")`` runs the local model as the teacher.
    Host override: ``FEDDAPT_OLLAMA_HOST`` (default http://localhost:11434)."""
    low = model.lower()
    for pre in _LOCAL_PREFIXES:
        if low.startswith(pre):
            name = model[len(pre):]                       # keep model-name case
            host = os.environ.get("FEDDAPT_OLLAMA_HOST", "http://localhost:11434")
            return _ollama_llm(name, temperature, host)

    anthropic_key = _api_key("ANTHROPIC_API_KEY")
    if "claude" in model.lower() or anthropic_key:
        import anthropic
        if not anthropic_key:
            raise RuntimeError(
                f"ANTHROPIC_API_KEY is not set, needed for judge model {model!r}. "
                "Put it in .env (see .env.example) or export it in your shell."
            )
        client = anthropic.Anthropic(api_key=anthropic_key)
        kwargs = {} if model.startswith(_NO_TEMPERATURE) else {"temperature": temperature}

        def call(prompt: str) -> str:
            resp = client.messages.create(
                model=model, max_tokens=1024, **kwargs,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text
        return _with_retry(call)

    from openai import OpenAI
    openai_key = _api_key("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set, needed for judge model {model!r}. "
            "Put it in .env (see .env.example) or export it in your shell."
        )
    client = OpenAI(api_key=openai_key)

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    return _with_retry(call)


def _verdict(text: str) -> bool:
    return "correct" in text.lower().split("verdict:", 1)[-1][:20] and \
           "incorrect" not in text.lower().split("verdict:", 1)[-1][:20]


def judge_one(llms: list[Callable[[str], str]], question, candidate, reference):
    """CLEV vote: two primary judges; third only on disagreement. Falls back to
    a single judge if only one model is supplied (with a validity warning)."""
    prompt = JUDGE_PROMPT.format(q=question, ref=reference, cand=candidate)
    if len(llms) == 1:                                  # single-judge (less reliable)
        out = llms[0](prompt)
        return _verdict(out), [out]
    outs = [llms[0](prompt), llms[1](prompt)]
    v = [_verdict(o) for o in outs]
    if v[0] != v[1] and len(llms) > 2:                  # tie-break only on disagreement
        outs.append(llms[2](prompt)); v.append(_verdict(outs[-1]))
    return sum(v) > len(v) / 2, outs


def score_free_form(llms, examples) -> float:
    """Fraction judged correct. `examples` = [{question, candidate, reference}]."""
    if not examples:
        return 0.0
    ok = sum(judge_one(llms, e["question"], e["candidate"], e["reference"])[0]
             for e in examples)
    return ok / len(examples)


def validate_judge(llms, human_labeled) -> dict:
    """Compare judge verdicts to human labels. human_labeled adds 'human' (bool).
    Returns Cohen's kappa + macro-F1 — admit the judge only if it clears the bar
    (CLEV: kappa>=0.6, F1>=0.85)."""
    from sklearn.metrics import cohen_kappa_score, f1_score
    y_h, y_j = [], []
    for e in human_labeled:
        y_h.append(bool(e["human"]))
        y_j.append(judge_one(llms, e["question"], e["candidate"], e["reference"])[0])
    return {"cohen_kappa": float(cohen_kappa_score(y_h, y_j)),
            "macro_f1": float(f1_score(y_h, y_j, average="macro", zero_division=0)),
            "n": len(human_labeled)}
