"""Pure-function tests for the analysis layer (no matplotlib needed)."""
from fedapt.analysis import headline_metric, _epsilon


def test_headline_prefers_judge_then_f1_then_rouge():
    assert headline_metric({"verdict": {"judge_correct": 0.9, "verdict_macro_f1": 0.5}}) == 0.9
    assert headline_metric({"verdict": {"verdict_macro_f1": 0.7}}) == 0.7
    assert abs(headline_metric({"explain_log": {"rouge_l": 0.4},
                                "general_qa": {"rouge_l": 0.6}}) - 0.5) < 1e-9
    assert headline_metric({}) == 0.0


def test_epsilon_parsing():
    assert _epsilon({"epsilon": "inf"}) == float("inf")
    assert _epsilon({"epsilon": 3}) == 3.0
    assert _epsilon({}) == float("inf")
