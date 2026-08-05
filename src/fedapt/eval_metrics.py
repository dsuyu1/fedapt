"""Reference-based metrics — the SECONDARY signal.

ROUGE-L and BERTScore measure surface / embedding overlap with a reference.
They are cheap and reported, but (per CLEV, Badshah et al. 2025) they undercount
semantic equivalence — so the PRIMARY correctness signal is the LLM judge
(see fedapt.judge), not these.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _rouge():
    import evaluate
    return evaluate.load("rouge")


def rouge_l(predictions, references) -> float:
    if not predictions:
        return 0.0
    return float(_rouge().compute(predictions=list(predictions),
                                  references=list(references))["rougeL"])


def bert_score(predictions, references, lang="en") -> float:
    if not predictions:
        return 0.0
    from bert_score import score
    _, _, f1 = score(list(predictions), list(references), lang=lang, verbose=False)
    return float(f1.mean())


def parse_verdict(text: str) -> str | None:
    """Extract the capture-level label from a paragraph verdict
    ('Assessment: attack' / 'Assessment: benign'). 'malicious' is accepted as a
    synonym for 'attack' for robustness."""
    low = text.lower()
    is_attack = lambda s: "attack" in s or "malicious" in s
    if "assessment:" in low:
        tail = low.split("assessment:", 1)[1][:30]
        if is_attack(tail):
            return "attack"
        if "benign" in tail:
            return "benign"
    if is_attack(low) and "benign" not in low:
        return "attack"
    if "benign" in low and not is_attack(low):
        return "benign"
    return None
