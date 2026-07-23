"""LLM-as-a-judge — the PRIMARY correctness signal for free-form outputs.

Implements CLEV (Badshah et al. 2025, arXiv:2503.08542): reference-aware judging
that returns a binary verdict + rationale, with lightweight voting — two judges
decide, a third breaks ties only on disagreement (~80-95% cheaper than a fixed
3-judge panel).

Integrity rules (DESIGN.md §8), enforced by convention here:
  * the judge model MUST differ from the base model under test (no self-judging);
  * validate the judge against a small human-labelled set (`validate_judge`)
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


def make_llm(model: str, temperature: float = 0.0) -> Callable[[str], str]:
    """Return a text->text callable backed by an LLM API (Anthropic or OpenAI)."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if "claude" in model.lower() or anthropic_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)

        def call(prompt: str) -> str:
            resp = client.messages.create(
                model=model, max_tokens=1024, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text
        return call

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    return call


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
