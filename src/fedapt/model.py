"""Base model + LoRA (QLoRA) loading and LoRA-state plumbing.

Kept tiny on purpose: the federated code moves LoRA *state dicts* around, so the
only model-specific things it needs are "give me a fresh adapter" and "get/set
the LoRA weights". Requires the `train` extra (torch/transformers/peft/bnb).
"""
from __future__ import annotations

from .config import Config


def load_base_with_lora(cfg: Config):
    """Return a fresh 4-bit base model with an untrained LoRA adapter attached."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, quantization_config=bnb, device_map="auto")
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      target_modules=["q_proj", "v_proj"], lora_dropout=cfg.lora_dropout,
                      bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base, lora)


def get_lora_state(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora" in n}


def set_lora_state(model, state: dict) -> None:
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(p.dtype).to(p.device))


def state_to_arrays(state: dict):
    keys = sorted(state.keys())
    return [state[k].numpy() for k in keys], keys


def arrays_to_state(arrays, keys) -> dict:
    import torch
    return {k: torch.tensor(v) for k, v in zip(keys, arrays)}


def perplexity(model, tokenizer, texts, max_length, device, max_docs=200) -> float:
    """Held-out raw-text perplexity — the DAPT convergence/validation signal."""
    import numpy as np, torch
    was_training = model.training
    model.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for t in texts[:max_docs]:
            ids = tokenizer(t, truncation=True, max_length=max_length,
                            return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            loss = model(input_ids=ids, labels=ids).loss
            nll += float(loss) * (ids.shape[1] - 1); ntok += ids.shape[1] - 1
    if was_training:
        model.train()
    return float(np.exp(nll / max(ntok, 1)))


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_packed_dataset(tokenizer, texts, max_length):
    """Pack short docs end-to-end (EOS-separated) into fixed-length blocks so no
    GPU step is wasted on padding — security docs average ~65 words."""
    from datasets import Dataset
    ids = []
    for t in texts:
        ids += tokenizer(t, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    chunks = [ids[i:i + max_length] for i in range(0, len(ids), max_length)]
    rows = [{"input_ids": c, "attention_mask": [1] * len(c), "labels": c[:]}
            for c in chunks if len(c) >= 16]
    return Dataset.from_list(rows)


def save_adapter(model, path, meta: dict):
    import json, os
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)                       # PEFT adapter weights
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=2)
    print("saved adapter:", path)


def _bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)


def load_task_model(cfg: Config, init_from: str | None = None):
    """Base + LoRA for Stage-2 tuning. If `init_from` (an adapter id under
    adapters/) is given, warm-start from it (trainable); else a fresh adapter.
    `init_from=None` is the no-DAPT ablation; a DAPT adapter id is A/B/C/D."""
    import os
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, quantization_config=_bnb_config(),
                                                device_map="auto")
    if init_from:
        from peft import PeftModel
        return PeftModel.from_pretrained(base, os.path.join(cfg.adapters_dir, init_from),
                                         is_trainable=True)
    from peft import LoraConfig, get_peft_model
    return get_peft_model(base, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=["q_proj", "v_proj"],
        lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM"))


def load_eval_model(cfg: Config, adapter_id: str | None = None):
    """Base (+ optional adapter) for inference only. `adapter_id=None` = zero-shot."""
    import os
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, quantization_config=_bnb_config(),
                                                device_map="auto")
    if adapter_id:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, os.path.join(cfg.adapters_dir, adapter_id))
    base.eval()
    return base
