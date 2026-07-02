from __future__ import annotations

import os
from typing import Mapping


def loss_weights_from_env(prefix: str, defaults: Mapping[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key, default in defaults.items():
        env_name = f"{prefix}_LOSS_{key.upper()}"
        alias_name = None
        if key.startswith("loss_"):
            alias_name = f"{prefix}_LOSS_{key.removeprefix('loss_').upper()}"

        value = os.getenv(env_name)
        if value is None and alias_name is not None:
            value = os.getenv(alias_name)
        weights[key] = float(value) if value is not None else default
    return weights


def compute_weighted_loss(
    loss_dict: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
):
    if not loss_dict:
        raise ValueError("loss_dict vazio; não há losses para combinar.")

    if not weights:
        return sum(loss for loss in loss_dict.values())

    weighted_losses = [
        loss * float(weights.get(name, 1.0))
        for name, loss in loss_dict.items()
    ]
    return sum(weighted_losses)


__all__ = ["compute_weighted_loss", "loss_weights_from_env"]
