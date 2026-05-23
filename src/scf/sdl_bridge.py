"""
Bridge to the sibling `sparse-dictionary-learning` clone.

We don't fork that repo. We add its `transformer/` and `autoencoder/` to
sys.path on demand so we can import `HookedGPT`, `GPTConfig`, and `AutoEncoder`.
"""
from __future__ import annotations
import os
import sys
from typing import Tuple

import torch

from .config import Config


def _ensure_sdl_on_path(sdl_repo: str) -> None:
    if not os.path.isdir(sdl_repo):
        raise FileNotFoundError(
            f"sdl_repo not found: {sdl_repo}. Set `sdl_repo` in configs/default.yaml."
        )
    for sub in ("transformer", "autoencoder"):
        p = os.path.join(sdl_repo, sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def load_transformer(cfg: Config) -> Tuple["torch.nn.Module", int]:
    """Return (HookedGPT, d_mlp). Requires `gpt_ckpt_dir/ckpt.pt` to exist."""
    _ensure_sdl_on_path(cfg.sdl_repo)
    from model import GPTConfig          # type: ignore
    from hooked_model import HookedGPT   # type: ignore

    ckpt_path = os.path.join(cfg.sdl_repo, "transformer", cfg.gpt_ckpt_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Transformer checkpoint missing: {ckpt_path}.\n"
            "Train it on HPC first: see docs/HPC_PLAYBOOK.md, step 1."
        )

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    gpt_conf = GPTConfig(**ckpt["model_args"])
    model = HookedGPT(gpt_conf)
    sd = ckpt["model"]
    prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(cfg.device)
    d_mlp = 4 * gpt_conf.n_embd
    return model, d_mlp


def load_sae(cfg: Config, d_mlp: int) -> "torch.nn.Module":
    """Return a loaded AutoEncoder. Requires `sae_ckpt_subdir` to be set."""
    _ensure_sdl_on_path(cfg.sdl_repo)
    from autoencoder import AutoEncoder  # type: ignore

    if not cfg.sae_ckpt_subdir:
        raise ValueError(
            "configs/default.yaml `sae_ckpt_subdir` is empty. "
            "Train SAE on HPC and set the path."
        )
    sae_path = os.path.join(
        cfg.sdl_repo, "autoencoder", "out", cfg.dataset, cfg.sae_ckpt_subdir, "ckpt.pt"
    )
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE checkpoint missing: {sae_path}")

    ckpt = torch.load(sae_path, map_location=cfg.device)
    sd = ckpt["autoencoder"]
    n_features, n_in = sd["encoder.weight"].shape
    if n_in != d_mlp:
        raise ValueError(f"SAE expects d_mlp={n_in}, transformer gives {d_mlp}")
    sae = AutoEncoder(n_inputs=n_in, n_latents=n_features)
    sae.load_state_dict(sd)
    sae.eval().to(cfg.device)
    return sae


def get_tokenizer():
    """OpenWebText was trained with GPT-2 tokenizer."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")
