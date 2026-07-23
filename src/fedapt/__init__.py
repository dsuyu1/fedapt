"""FeDAPT — federated domain-adaptive pre-training for security assistant LLMs.

Layout (see DESIGN.md):
    config       one dataclass of all knobs + seeds + paths
    corpus       build the PROSE DAPT corpus (shared knowledge)
    clients      build LOG client datasets + non-IID split (private grounding)
    tasks        the 4 task builders + teacher target synthesis
    splits       train/val/test + held-out ids
    model        base + LoRA (QLoRA) loading
    federated    FedAvg/FedProx loop + validation-based round selection
    aggregators  fedavg / krum / trimmed_mean
    attacks      sign_flip / scale / gaussian / data_poison
    dp           DP-FedAvg (+ TODO real accountant)
    eval_metrics ROUGE / BERTScore
    judge        CLEV LLM-as-judge + human-validation harness
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
