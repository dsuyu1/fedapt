"""Federated DAPT training loop — Stage 1.

Many organizations each pre-train on their own
prose (like their documentation), and their LoRA updates are combined into one shared adapter — the
raw text never leaves an org, only the weight updates do. It composes the
already-built pieces:

    model.py        get/set LoRA state, packed datasets, perplexity
    aggregators.py  how updates are combined (fedavg / krum / trimmed_mean)
    attacks.py      how a malicious client corrupts its update / data
    dp.py           clip + noise on each update (privacy budget spent here)

Read `federated_dapt` top to bottom — it is the whole algorithm.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from . import model as M
from .aggregators import AGGREGATORS
from .attacks import poison_update, poison_dataset, UPDATE_ATTACKS
from .dp import noise_multiplier_for, achieved_epsilon, privatize


# --------------------------------------------------------------------------- #
# an experiment = one point in the matrix (which aggregator, DP level, attack)
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentSpec:
    exp_id: str
    aggregator: str = "fedavg"          # key into aggregators.AGGREGATORS
    epsilon: float = float("inf")       # inf = no DP
    mu: float = 0.0                     # FedProx strength (0 = FedAvg)
    malicious: frozenset = field(default_factory=frozenset)  # client indices
    attack: str = "none"                # none | sign_flip | scale | gaussian | data_poison


def default_matrix(cfg: Config) -> list[ExperimentSpec]:
    """FedAvg/FedProx + DP sweep + the Byzantine block (attack × aggregator)."""
    exps = [
        ExperimentSpec("dapt_fedavg_no_dp"),
        ExperimentSpec("dapt_fedprox_mu_0p01", mu=0.01),
        ExperimentSpec("dapt_fedavg_eps_8", epsilon=8),
        ExperimentSpec("dapt_fedavg_eps_3", epsilon=3),
        ExperimentSpec("dapt_fedavg_eps_1", epsilon=1),
    ]
    mal = frozenset(range(cfg.n_malicious))
    for atk in ("sign_flip", "scale", "gaussian", "data_poison"):
        for agg in ("fedavg", "krum", "trimmed_mean"):
            exps.append(ExperimentSpec(f"byz_{agg}_{atk}", aggregator=agg,
                                       malicious=mal, attack=atk))
    return exps


# --------------------------------------------------------------------------- #
# one client's local update
# --------------------------------------------------------------------------- #
def _local_train(model, dataset, cfg, global_state, use_fedprox, mu, device) -> float:
    """Train the (shared) model on ONE client's data for `local_steps` steps,
    in place. Returns the mean loss. This is the only place gradients flow."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    model.train()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=default_data_collator)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.learning_rate)
    # FedProx keeps the local model from drifting too far from the global one on
    # non-IID data: add (mu/2)*||w - w_global||^2 to the loss.
    gref = {n: global_state[n].to(device) for n in global_state} if use_fedprox else None

    losses, step = [], 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        if use_fedprox:
            prox = sum(((p - gref[n]) ** 2).sum()
                       for n, p in model.named_parameters() if n in gref)
            loss = loss + (mu / 2.0) * prox
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(float(loss)); step += 1
        if step >= cfg.local_steps:
            break
    return float(np.mean(losses)) if losses else 0.0


# --------------------------------------------------------------------------- #
# the federated loop
# --------------------------------------------------------------------------- #
def federated_dapt(cfg: Config, client_prose: dict, spec: ExperimentSpec,
                   lm_val: list, model=None, tokenizer=None):
    """Run one federated DAPT experiment.

    Returns (best_lora_state, round_log). The model is left holding the best
    round's weights (selected by validation perplexity).
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = M.load_base_with_lora(cfg)
    if tokenizer is None:
        tokenizer = M.load_tokenizer(cfg)

    agg_fn = AGGREGATORS[spec.aggregator]
    use_dp = math.isfinite(spec.epsilon)
    # RDP accountant: turn the TARGET epsilon into the noise multiplier for this
    # many rounds at full participation (falls back to the crude map if offline).
    noise_mult = noise_multiplier_for(spec.epsilon, cfg.dp_delta, cfg.num_rounds,
                                      cfg.dp_sample_rate, fallback_map=cfg.dp_noise_map)

    # Fixed client order so `malicious` indices are stable across rounds.
    names = sorted(client_prose.keys())
    datasets = {n: M.build_packed_dataset(tokenizer, client_prose[n], cfg.max_seq_length)
                for n in names}
    # data_poison attackers train on scrambled tokens; precompute those sets once.
    poisoned = {}
    if spec.attack == "data_poison":
        for ci, n in enumerate(names):
            if ci in spec.malicious:
                poisoned[n] = poison_dataset(datasets[n], seed=cfg.seed)

    # The shared model state that gets passed around and refined each round.
    global_state = M.get_lora_state(model)
    keys = sorted(global_state.keys())
    best_ppl, best_state, log = float("inf"), None, []

    print(f"--- {spec.exp_id} | agg={spec.aggregator} eps={spec.epsilon} "
          f"mu={spec.mu} dp={use_dp} malicious={sorted(spec.malicious)} attack={spec.attack} ---")

    for r in range(cfg.num_rounds):
        g_arrays, _ = M.state_to_arrays(global_state)   # current global as flat arrays
        client_updates, sizes, round_losses = [], [], []

        # ---- each org trains locally on its own shard, starting from global ----
        for ci, n in enumerate(names):
            is_mal = ci in spec.malicious
            ds = poisoned[n] if (is_mal and spec.attack == "data_poison") else datasets[n]

            M.set_lora_state(model, global_state)       # reset to the shared model
            loss = _local_train(model, ds, cfg, global_state, spec.mu > 0, spec.mu, device)
            new_arrays, _ = M.state_to_arrays(M.get_lora_state(model))  # this org's update

            # a malicious org corrupts its update before sending it
            if is_mal and spec.attack in UPDATE_ATTACKS:
                new_arrays = poison_update(new_arrays, g_arrays, spec.attack,
                                           cfg.byz_boost, seed=r * 100 + ci)
            # differential privacy: clip + noise the update (privacy spent HERE)
            if use_dp:
                delta = [na - ga for na, ga in zip(new_arrays, g_arrays)]
                delta = privatize(delta, cfg.dp_max_grad_norm, noise_mult, seed=r * 100 + ci)
                new_arrays = [ga + d for ga, d in zip(g_arrays, delta)]

            client_updates.append(new_arrays)
            sizes.append(len(ds))
            round_losses.append(loss)

        # ---- server combines the updates into a new shared model ----
        global_state = M.arrays_to_state(agg_fn(client_updates, sizes), keys)

        # ---- validation: perplexity on held-out prose; keep the best round ----
        M.set_lora_state(model, global_state)
        ppl = M.perplexity(model, tokenizer, lm_val, cfg.max_seq_length, device) if lm_val else float("nan")
        log.append({"round": r + 1, "mean_loss": float(np.mean(round_losses)), "val_ppl": ppl})
        if lm_val and ppl < best_ppl:
            best_ppl = ppl
            best_state = {k: v.clone() for k, v in global_state.items()}
        print(f"  round {r + 1}/{cfg.num_rounds} loss={np.mean(round_losses):.4f} val_ppl={ppl:.2f}")

    final = best_state if best_state is not None else global_state
    M.set_lora_state(model, final)
    if best_state is not None:
        print(f"  kept best round by val_ppl = {best_ppl:.2f}")
    return final, log


# --------------------------------------------------------------------------- #
# driver — run the whole matrix, save one adapter per experiment
# --------------------------------------------------------------------------- #
def run_matrix(cfg: Config, specs: list[ExperimentSpec] | None = None):
    cfg.ensure_dirs()
    client_prose = {
        os.path.basename(p).replace("_prose.json", ""): json.load(open(p))
        for p in sorted(glob.glob(os.path.join(cfg.clients_dir, "client_*_prose.json")))
    }
    if not client_prose:
        raise SystemExit("no prose client shards — run scripts/build_data.py first")

    lm_val_path = os.path.join(cfg.eval_dir, "lm_val.json")
    lm_val = json.load(open(lm_val_path)) if os.path.exists(lm_val_path) else []

    for spec in (specs or default_matrix(cfg)):
        path = os.path.join(cfg.adapters_dir, spec.exp_id)
        if os.path.exists(os.path.join(path, "meta.json")):
            print("skip (done):", spec.exp_id); continue
        model = M.load_base_with_lora(cfg)
        tok = M.load_tokenizer(cfg)
        best_state, round_log = federated_dapt(cfg, client_prose, spec, lm_val, model, tok)
        M.set_lora_state(model, best_state)
        # honest privacy accounting: record σ used and the ε actually spent
        eps_spent, noise_mult = None, 0.0
        if math.isfinite(spec.epsilon):
            noise_mult = noise_multiplier_for(spec.epsilon, cfg.dp_delta, cfg.num_rounds,
                                              cfg.dp_sample_rate, fallback_map=cfg.dp_noise_map)
            try:
                eps_spent = achieved_epsilon(noise_mult, cfg.dp_delta, cfg.num_rounds, cfg.dp_sample_rate)
            except Exception:
                eps_spent = None                      # opacus missing -> can't certify
        M.save_adapter(model, path, {
            "id": spec.exp_id, "stage": "dapt", "aggregator": spec.aggregator,
            "epsilon": ("inf" if not math.isfinite(spec.epsilon) else spec.epsilon),
            "epsilon_target": ("inf" if not math.isfinite(spec.epsilon) else spec.epsilon),
            "epsilon_spent": eps_spent, "noise_multiplier": noise_mult, "dp_delta": cfg.dp_delta,
            "mu": spec.mu, "attack": spec.attack, "num_malicious": len(spec.malicious),
            "round_log": round_log,
        })
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("matrix done")


def run_local_dapt(cfg: Config):
    """Ablation B: each org does DAPT ALONE (no aggregation) on its own prose
    shard. Compute budget matches the federated runs (num_rounds*local_steps).
    Saves one `dapt_local_<client>` adapter per org for Stage-2 warm-starting."""
    import torch
    cfg.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = M.load_tokenizer(cfg)
    shards = {
        os.path.basename(p).replace("_prose.json", ""): json.load(open(p))
        for p in sorted(glob.glob(os.path.join(cfg.clients_dir, "client_*_prose.json")))
    }
    budget = cfg.num_rounds * cfg.local_steps
    for client, texts in shards.items():
        out = os.path.join(cfg.adapters_dir, f"dapt_local_{client}")
        if os.path.exists(os.path.join(out, "meta.json")):
            print("skip (done):", client); continue
        model = M.load_base_with_lora(cfg)
        ds = M.build_packed_dataset(tok, texts, cfg.max_seq_length)
        state = M.get_lora_state(model)                 # global_state unused (mu=0)
        steps_done = 0
        while steps_done < budget:                      # keep training until budget met
            _local_train(model, ds, cfg, state, use_fedprox=False, mu=0.0, device=device)
            steps_done += cfg.local_steps
        M.save_adapter(model, out, {"id": f"dapt_local_{client}", "stage": "dapt",
                                    "aggregator": "local", "epsilon": "inf", "mu": 0.0,
                                    "attack": "none", "num_malicious": 0, "client": client})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("local DAPT adapters ready (ablation B)")
