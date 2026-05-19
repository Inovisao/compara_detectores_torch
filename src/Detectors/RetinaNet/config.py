from __future__ import annotations

import os
from dataclasses import asdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RetinaNetConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    lr_step_size: int
    lr_gamma: float
    num_workers: int
    device: str | None


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_config() -> RetinaNetConfig:
    return RetinaNetConfig(
        epochs=_get_env_int("RETINANET_EPOCHS", 1000),
        batch_size=_get_env_int("RETINANET_BATCH", 4),
        learning_rate=_get_env_float("RETINANET_LR", 1e-4),
        momentum=_get_env_float("RETINANET_MOMENTUM", 0.9),
        weight_decay=_get_env_float("RETINANET_WEIGHT_DECAY", 1e-4),
        lr_step_size=_get_env_int("RETINANET_LR_STEP", 5),
        lr_gamma=_get_env_float("RETINANET_LR_GAMMA", 0.1),
        num_workers=_get_env_int("RETINANET_NUM_WORKERS", 4),
        device=os.getenv("RETINANET_DEVICE"),
    )


def get_training_params() -> dict:
    return asdict(get_config())


__all__ = ["RetinaNetConfig", "get_config", "get_training_params"]
