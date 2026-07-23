"""Stage 2 — local task tuning (optional, per organisation).

Each org warm-starts from a Stage-1 adapter and tunes on ITS OWN log-grounded
task pairs. The data never leaves the org (that's the whole point), so this runs
per client and we macro-average at eval time.

The starting point is the ablation variable — same tuning, different init:
    A  init=None                    no DAPT
    B  init=dapt_local_<client>     the org's own local DAPT   (needs run_local_dapt)
    C  init=dapt_fedavg_no_dp       the shared federated DAPT  (headline)
    D  init=dapt_centralized        centralized ceiling        (optional)

Only the log-grounded tasks are tuned per org (`explain_log`, `verdict`); the
prose tasks are general knowledge already covered by DAPT and are eval-only.
"""
from __future__ import annotations

import glob
import json
import os

from .config import Config
from . import model as M

# tasks that are org-private (log-grounded) and therefore tuned locally
LOG_TASKS = ["explain_log", "verdict"]

# default ablation rows: label -> init spec ('LOCAL' expands per client)
DEFAULT_ROWS = {"A": None, "B": "LOCAL", "C": "dapt_fedavg_no_dp"}


def _instruction(rec) -> str:
    return f"### Instruction:\n{rec['input']}\n\n### Response:\n{rec['target']}"


def _client_train_records(cfg: Config, client: str) -> list[dict]:
    """This client's TRAIN records across the log-grounded tasks."""
    recs = []
    for t in LOG_TASKS:
        p = os.path.join(cfg.eval_dir, f"{t}_split.json")
        if not os.path.exists(p):
            continue
        recs += [r for r in json.load(open(p))["train"] if r.get("client") == client]
    return recs


def train_client(cfg: Config, init_from: str | None, row: str, client: str, tokenizer):
    out_id = f"task_{row}__{client}"
    out = os.path.join(cfg.adapters_dir, out_id)
    if os.path.exists(os.path.join(out, "meta.json")):
        print("skip (done):", out_id); return
    recs = _client_train_records(cfg, client)
    if not recs:
        print("  no task-train data for", client, "— skipping"); return

    import torch
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    model = M.load_task_model(cfg, init_from)

    def tok(batch):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        RESPONSE_HEADER = "\n\n### Response:\n"

        for rec in batch["raw"]:
            prompt = f"### Instruction:\n{rec['input']}{RESPONSE_HEADER}"
            target = f"{rec['target']}{tokenizer.eos_token}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

            full_ids = (prompt_ids + target_ids)[:cfg.max_seq_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            pad_len = cfg.max_seq_length - len(full_ids)

            attention_mask = [1] * len(full_ids) + [0] * pad_len
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            input_ids_list.append(full_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    ds = Dataset.from_list([{"raw": r} for r in recs]).map(
        tok, batched=True, remove_columns=["raw"])

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(cfg.scratch, out_id),
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=1, learning_rate=1e-5,            # low LR: continue an adapted adapter
            bf16=torch.cuda.is_available(), optim="paged_adamw_8bit",
            save_strategy="no", report_to="none", logging_steps=20),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ).train()

    M.save_adapter(model, out, {"id": out_id, "stage": "task", "row": row,
                                "init_from": init_from, "client": client})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_task_tuning(cfg: Config, rows: dict | None = None):
    """Run the ablation rows over every client. Produces task_<row>__<client>."""
    cfg.ensure_dirs()
    tokenizer = M.load_tokenizer(cfg)
    clients = sorted(
        os.path.basename(p).replace("_prose.json", "")
        for p in glob.glob(os.path.join(cfg.clients_dir, "client_*_prose.json"))
    )
    for row, init in (rows or DEFAULT_ROWS).items():
        for client in clients:
            init_from = f"dapt_local_{client}" if init == "LOCAL" else init
            print(f"== row {row} | {client} | init={init_from} ==")
            train_client(cfg, init_from, row, client, tokenizer)
    print("task tuning done")
