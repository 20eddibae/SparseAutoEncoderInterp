"""
ShareGPT loader. Expects a local JSON file in the format
[{"conversations": [{"from": "human"|"gpt", "value": "..."}, ...]}, ...].

We don't auto-download — the file is ~700MB. See docs/HPC_PLAYBOOK.md for
where to fetch it.
"""
from __future__ import annotations
import json
import os
from typing import Iterable

from ..config import Config
from .types import Conversation, Turn

_ROLE_MAP = {"human": "human", "user": "human", "gpt": "ai", "assistant": "ai"}


def iter_sharegpt(cfg: Config) -> Iterable[Conversation]:
    path = cfg.corpus.sharegpt_path
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ShareGPT file not found at {path}. See docs/HPC_PLAYBOOK.md."
        )
    with open(path) as f:
        data = json.load(f)

    yielded = 0
    for i, item in enumerate(data):
        raw_turns = item.get("conversations") or item.get("messages") or []
        turns: list[Turn] = []
        for t in raw_turns[: cfg.corpus.max_turns]:
            role_in = (t.get("from") or t.get("role") or "").lower()
            role = _ROLE_MAP.get(role_in)
            text = t.get("value") or t.get("content") or ""
            if role is None or not text.strip():
                continue
            turns.append(Turn(role=role, text=text))
        if len(turns) < 2:
            continue
        yield Conversation(turns=turns, meta={"source": "sharegpt", "idx": i})
        yielded += 1
        if yielded >= cfg.corpus.n_conversations:
            break
