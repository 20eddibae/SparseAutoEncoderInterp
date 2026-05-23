"""
Turn a text string into a per-turn SAE-feature vector.

Pipeline:
  text -> tokens -> HookedGPT forward (stores last MLP GELU output) ->
  AutoEncoder.encode -> (token, feature) activation map ->
  pool over tokens -> 1-D activation vector of length n_features ->
  threshold -> binary support set.

We return both the float magnitudes and the binary support. Downstream code
chooses which to use (entropy/CLT need magnitudes; Markov chains need support).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import Config
from .sdl_bridge import load_transformer, load_sae, get_tokenizer


@dataclass
class TurnFeatures:
    magnitudes: np.ndarray   # shape (n_features,), float32
    support: np.ndarray      # shape (n_features,), bool
    n_active: int


class FeatureExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer = get_tokenizer()
        self.transformer, self.d_mlp = load_transformer(cfg)
        self.sae = load_sae(cfg, self.d_mlp)
        self.n_features = self.sae.n_latents
        self.block_size = self.transformer.config.block_size

    @torch.no_grad()
    def encode_turn(self, text: str) -> TurnFeatures:
        ids = self.tokenizer.encode_ordinary(text)
        if len(ids) == 0:
            empty = np.zeros(self.n_features, dtype=np.float32)
            return TurnFeatures(empty, empty.astype(bool), 0)

        max_t = min(self.cfg.max_tokens_per_turn, self.block_size)
        ids = ids[-max_t:]
        x = torch.tensor(ids, dtype=torch.long, device=self.cfg.device).unsqueeze(0)

        _ = self.transformer(x)  # populates self.transformer.mlp_activation_hooks
        mlp_acts = self.transformer.mlp_activation_hooks[-1]  # (1, T, d_mlp)
        self.transformer.clear_mlp_activation_hooks()
        latents = self.sae.encode(mlp_acts.squeeze(0))        # (T, n_features)

        mags = self._pool(latents).cpu().numpy().astype(np.float32)
        support = mags > self.cfg.activation_threshold
        return TurnFeatures(mags, support, int(support.sum()))

    def _pool(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (T, F). Returns (F,)."""
        if self.cfg.turn_pooling == "max":
            return latents.amax(dim=0)
        if self.cfg.turn_pooling == "mean":
            return latents.mean(dim=0)
        if self.cfg.turn_pooling == "any":
            return (latents > self.cfg.activation_threshold).any(dim=0).float()
        raise ValueError(f"unknown turn_pooling: {self.cfg.turn_pooling}")
