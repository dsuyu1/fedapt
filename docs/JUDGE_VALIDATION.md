# Validating the LLM Judge (human audit)

You must show the CLEV judge agrees with humans before you trust its scores.
This is a **one-time** check on the judge, done on ~50 hand-labeled items.

## The four roles (don't mix them up)
- **Student** — the model under test. Produces the **candidate** answer.
- **Teacher** — synthesized the **reference** (gold) answers.
- **Judge** — the independent LLM (Claude Haiku/Sonnet) that says correct/incorrect.
- **You (human)** — produce the **ground-truth labels** the judge is scored against.

You are validating the **judge**. The student/teacher outputs are just the material.
You are NOT auditing the teacher here, and NOT rubber-stamping the judge — you
label independently, then compare.

## Steps
1. **Generate the sheet** (student model; GPU; no Anthropic):
   ```
   python scripts/make_judge_audit.py --n 50 --mix dapt_fedavg_no_dp
   ```
   `--mix` = half candidates from the zero-shot base, half from the adapter, so you
   get both correct and incorrect answers. Writes `FEDDAPT_ROOT/eval/judge_audit.csv`.

2. **Label by hand** (free; no compute). Open the CSV, fill the **`human`** column:
   `1` = candidate is factually correct vs the reference, `0` = not.
   **Label BLIND** — decide yourself; never look at what the judge would say
   (otherwise you bias toward agreeing with it). Aim for ~50 rows.

3. **Score the judge** (needs Anthropic; run when credits return):
   ```
   python scripts/score_judge.py
   ```
   Prints Cohen's κ + macro-F1 of judge-vs-you. **Admit the judge only if
   κ ≥ 0.60 and macro-F1 ≥ 0.85** (the CLEV bar). If it fails, use a stronger
   judge model or refine the prompt and re-score.

## Report in the paper
State the judge model(s), the number of human-labeled items, and the κ / macro-F1
you measured — that's what makes the LLM-judge metric defensible to reviewers.
