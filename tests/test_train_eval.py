"""Pure-Python tests for the Stage-2 / eval logic (no torch, no API)."""
from fedapt.train_tasks import _instruction, DEFAULT_ROWS
from fedapt.evaluate import _macro_average, PROMPTS
from fedapt.judge import _verdict


def test_instruction_format_roundtrips_input_and_target():
    rec = {"input": "sysmon: powershell -enc ...", "target": "This is encoded PowerShell."}
    s = _instruction(rec)
    assert rec["input"] in s and rec["target"] in s
    assert s.index("### Instruction") < s.index("### Response")


def test_ablation_rows_defined():
    assert DEFAULT_ROWS["A"] is None
    assert DEFAULT_ROWS["B"] == "LOCAL"
    assert "dapt" in DEFAULT_ROWS["C"]


def test_prompt_templates_have_placeholder():
    for t, tmpl in PROMPTS.items():
        assert "{q}" in tmpl


def test_macro_average_across_clients():
    a = {"verdict": {"verdict_macro_f1": 0.8, "rouge_l": 0.5}}
    b = {"verdict": {"verdict_macro_f1": 0.6, "rouge_l": 0.7}}
    out = _macro_average([a, b])
    assert abs(out["verdict"]["verdict_macro_f1"] - 0.7) < 1e-9
    assert abs(out["verdict"]["rouge_l"] - 0.6) < 1e-9


def test_judge_verdict_parse():
    assert _verdict("Verdict: correct — matches the reference") is True
    assert _verdict("Verdict: incorrect, misses the point") is False
