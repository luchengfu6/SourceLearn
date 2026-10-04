"""Per-source train/test splits. One seed fixes BOTH membership and the
training order (M is updated sequentially, so order is part of the
experiment)."""
from __future__ import annotations

import random


def make_split(qids: list[str], seed: int, train_frac: float = 0.3, min_train: int = 4) -> dict:
    ids = sorted(qids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(round(train_frac * len(ids)))
    if n_train < min_train:           # too few questions to train on
        return {"seed": seed, "train": [], "test": sorted(ids), "trainable": False}
    return {"seed": seed, "train": ids[:n_train], "test": sorted(ids[n_train:]), "trainable": True}
