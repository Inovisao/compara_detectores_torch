from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ViTConfig:
    model_name: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
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


def get_config() -> ViTConfig:
    return ViTConfig(
        model_name=os.getenv("VIT_MODEL_NAME", "hustvl/yolos-small"),
        image_size=_get_env_int("VIT_IMAGE_SIZE", 640),
        epochs=_get_env_int("VIT_EPOCHS", 150),
        batch_size=_get_env_int("VIT_BATCH", 8),
        learning_rate=_get_env_float("VIT_LR", 1e-4),
        weight_decay=_get_env_float("VIT_WEIGHT_DECAY", 1e-4),
        lr_step_size=_get_env_int("VIT_LR_STEP", 50),
        lr_gamma=_get_env_float("VIT_LR_GAMMA", 0.1),
        num_workers=_get_env_int("VIT_NUM_WORKERS", 4),
        device=os.getenv("VIT_DEVICE"),
    )


def get_training_params() -> dict:
    return asdict(get_config())


__all__ = ["ViTConfig", "get_config", "get_training_params"]
