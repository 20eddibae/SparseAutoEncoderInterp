from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Literal

from ..config import Config

Role = Literal["human", "ai"]


@dataclass
class Turn:
    role: Role
    text: str


@dataclass
class Conversation:
    turns: List[Turn]
    meta: dict


def load_corpus(cfg: Config) -> Iterable[Conversation]:
    name = cfg.corpus.name
    if name == "sharegpt":
        from .sharegpt import iter_sharegpt
        return iter_sharegpt(cfg)
    if name == "synthetic":
        from .synthetic import iter_synthetic
        return iter_synthetic(cfg)
    raise ValueError(f"unknown corpus: {name}")
