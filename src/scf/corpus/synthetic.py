"""
Synthetic self-play: seed with a snippet from OpenWebText val.bin, then
alternate human/ai turns by sampling from the trained transformer.

This loader requires a working transformer checkpoint via sdl_bridge. Without
one it raises FileNotFoundError, like the other loaders.
"""
from __future__ import annotations
import os
from typing import Iterable, List

import numpy as np
import torch

from ..config import Config
from .types import Conversation, Turn


def _load_seeds(cfg: Config, n: int, seed_len: int = 32, rng_seed: int = 0) -> List[str]:
    from ..sdl_bridge import get_tokenizer
    enc = get_tokenizer()
    bin_path = os.path.join(
        cfg.sdl_repo, "transformer", "data", cfg.dataset, "val.bin"
    )
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"{bin_path} missing. Run sdl_repo/transformer/data/openwebtext/prepare.py."
        )
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(rng_seed)
    seeds: List[str] = []
    for _ in range(n):
        start = int(rng.integers(0, max(1, data.shape[0] - seed_len)))
        ids = data[start : start + seed_len].tolist()
        seeds.append(enc.decode(ids))
    return seeds


@torch.no_grad()
def _sample(model, enc, prompt_ids: List[int], max_new: int, device: str) -> List[int]:
    """Greedy-ish sampling: temperature=1.0, top-k=40. Simple, self-contained."""
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = list(prompt_ids)
    block = model.config.block_size
    for _ in range(max_new):
        x_cond = x if x.size(1) <= block else x[:, -block:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / 1.0
        topk = torch.topk(logits, 40)
        probs = torch.softmax(topk.values, dim=-1)
        idx = topk.indices[0, torch.multinomial(probs[0], 1)]
        out.append(int(idx.item()))
        x = torch.cat([x, idx.view(1, 1)], dim=1)
    return out


def iter_synthetic(cfg: Config) -> Iterable[Conversation]:
    from ..sdl_bridge import load_transformer, get_tokenizer

    model, _ = load_transformer(cfg)
    enc = get_tokenizer()
    seeds = _load_seeds(cfg, cfg.corpus.n_conversations)

    for i, seed_text in enumerate(seeds):
        turns: List[Turn] = [Turn(role="human", text=seed_text)]
        ctx_ids = enc.encode_ordinary(seed_text)
        for t in range(1, cfg.corpus.max_turns):
            role = "ai" if t % 2 == 1 else "human"
            new_ids = _sample(model, enc, ctx_ids, max_new=64, device=cfg.device)
            reply_ids = new_ids[len(ctx_ids):]
            turns.append(Turn(role=role, text=enc.decode(reply_ids)))
            ctx_ids = new_ids
        yield Conversation(turns=turns, meta={"source": "synthetic", "idx": i})
