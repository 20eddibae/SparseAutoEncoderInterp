from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class CoarsenConfig:
    method: str = "topk"
    k: int = 5
    n_clusters: int = 50


@dataclass
class CorpusConfig:
    name: str = "synthetic"
    n_conversations: int = 1000
    max_turns: int = 8
    sharegpt_path: str = "data/raw/sharegpt.json"
    # ShareGPT52K assistant turns are HTML-wrapped (<div class="markdown prose">),
    # human turns are plain. With strip_html=True the loader converts each turn's
    # value to plain text so role separability reflects language, not markup.
    # See scripts/evaluate_html_artifact.py for why this matters.
    strip_html: bool = False


@dataclass
class Config:
    sdl_repo: str = ""
    gpt_ckpt_dir: str = "out"
    sae_ckpt_subdir: str = ""
    dataset: str = "openwebtext"
    activation_threshold: float = 0.0
    turn_pooling: str = "max"
    device: str = "cpu"
    batch_size: int = 16
    max_tokens_per_turn: int = 512
    coarsen: CoarsenConfig = field(default_factory=CoarsenConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        raw = _expand_env_strings(raw)
        coarsen = CoarsenConfig(**raw.pop("coarsen", {}))
        corpus = CorpusConfig(**raw.pop("corpus", {}))
        cfg = cls(coarsen=coarsen, corpus=corpus, **raw)
        # SCF_SDL_REPO env var trumps the yaml. Useful when the same config
        # is consumed by the laptop and the cluster.
        env_override = os.environ.get("SCF_SDL_REPO")
        if env_override:
            cfg.sdl_repo = env_override
        return cfg


def _expand_env_strings(obj: Any) -> Any:
    """Recursively expand $HOME / ${HOME} in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    if isinstance(obj, dict):
        return {k: _expand_env_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_strings(v) for v in obj]
    return obj


def default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "configs", "default.yaml"))
