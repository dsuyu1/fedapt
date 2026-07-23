"""Single source of truth for all knobs, seeds, and paths.

Everything downstream takes a `Config`. Nothing reads globals or hardcodes a
path. This is what makes runs reproducible and the code explainable: to know
what a run did, read its Config.

Paths resolve like the old bootstrap did — Colab mounts Drive, otherwise use
`FEDDAPT_ROOT` (env var or .env). Secrets come from the environment / .env too.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- #
# environment helpers
# --------------------------------------------------------------------------- #
def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Real env vars take precedence."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if v:                                  # ignore blank entries (KEY=)
                    os.environ.setdefault(k.strip(), v)
    except FileNotFoundError:
        pass


def _env(name: str, default: str) -> str:
    """Env var, but a blank/unset value falls back to `default`."""
    return os.environ.get(name) or default


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _default_root() -> str:
    if _in_colab():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            pass
        return _env("FEDDAPT_ROOT", "/content/drive/MyDrive/FedDAPT")
    return _env("FEDDAPT_ROOT", os.path.abspath("./FedDAPT"))


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- paths (data lives outside git; see .gitignore) ---
    root: str = field(default_factory=_default_root)
    scratch: str = field(
        default_factory=lambda: _env(
            "FEDDAPT_WORK", os.path.join(tempfile.gettempdir(), "fedapt_work")
        )
    )
    # public attack_data checkout (malicious log client data); clone separately, it's LFS.
    attack_data_dir: str = field(
        default_factory=lambda: _env("FEDDAPT_ATTACK_DATA", "")
    )
    # benign telemetry dir (same log format) for the verdict task's negative class.
    benign_data_dir: str = field(
        default_factory=lambda: _env("FEDDAPT_BENIGN_DATA", "")
    )
    # vendor threat-intel / IR write-ups (.json/.txt) added to the DAPT prose corpus.
    vendor_data_dir: str = field(
        default_factory=lambda: _env("FEDDAPT_VENDOR_DATA", "")
    )

    # --- reproducibility ---
    seed: int = 42

    # --- model ---
    model_id: str = "mistralai/Mistral-7B-v0.1"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 512

    # --- federated clients / non-IID ---
    n_clients: int = 6
    dirichlet_alpha: float = 0.5          # smaller = more non-IID skew
    min_docs_per_client: int = 50

    # --- Stage 1: federated DAPT ---
    num_rounds: int = 20
    local_steps: int = 10
    learning_rate: float = 2e-5
    proximal_mu: float = 0.0              # FedProx; 0 = FedAvg

    # --- differential privacy (real RDP accountant; see dp.py) ---
    dp_max_grad_norm: float = 0.5
    dp_delta: float = 1e-5
    dp_sample_rate: float = 1.0           # full client participation each round
    dp_noise_map: dict = field(default_factory=lambda: {8: 0.5, 3: 0.8, 1: 1.2})  # offline fallback only

    # --- Byzantine / malicious clients ---
    n_malicious: int = 1
    byz_boost: float = 10.0               # sign-flip / scaling amplification
    trim_ratio: float = 0.2               # trimmed-mean (trims 1 each side at n=6)

    # --- data splits ---
    split_ratios: tuple = (0.70, 0.15, 0.15)
    lm_val_size: int = 300                # held-out raw docs for DAPT perplexity

    # --- evaluation / judge ---
    judge_model: str = field(
        default_factory=lambda: _env("FEDDAPT_JUDGE_MODEL", "claude-3-5-haiku-20241022")
    )
    judge_temperature: float = 0.0

    # ------------------------------------------------------------------ #
    # derived paths
    # ------------------------------------------------------------------ #
    @property
    def corpus_dir(self) -> str: return os.path.join(self.root, "corpus")
    @property
    def clients_dir(self) -> str: return os.path.join(self.root, "clients")
    @property
    def tasks_dir(self) -> str: return os.path.join(self.root, "tasks")
    @property
    def eval_dir(self) -> str: return os.path.join(self.root, "eval")
    @property
    def adapters_dir(self) -> str: return os.path.join(self.root, "adapters")
    @property
    def results_dir(self) -> str: return os.path.join(self.root, "results")
    @property
    def figures_dir(self) -> str: return os.path.join(self.root, "figures")

    def ensure_dirs(self) -> "Config":
        for d in (self.root, self.scratch, self.corpus_dir, self.clients_dir,
                  self.tasks_dir, self.eval_dir, self.adapters_dir,
                  self.results_dir, self.figures_dir):
            os.makedirs(d, exist_ok=True)
        return self

    def secret(self, name: str, default: str = "") -> str:
        """Read a secret from Colab userdata, else the environment / .env."""
        if _in_colab():
            try:
                from google.colab import userdata
                return (userdata.get(name) or default).strip()
            except Exception:
                pass
        return os.environ.get(name, default).strip()

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(**overrides) -> Config:
    """Load .env, build a Config, apply overrides, create dirs. Start here."""
    _load_dotenv()
    cfg = Config(**overrides)
    return cfg.ensure_dirs()
