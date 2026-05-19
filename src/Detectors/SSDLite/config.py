from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SSDLiteConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    momentum: float
    weight_decay: float
    lr_step_size: int
    lr_gamma: float
    num_workers: int
    score_thresh: float
    nms_thresh: float
    detections_per_img: int
    device: str | None


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_config() -> SSDLiteConfig:
    return SSDLiteConfig(
        epochs=_get_env_int("SSDLITE_EPOCHS", 200),
        batch_size=_get_env_int("SSDLITE_BATCH", 16),
        learning_rate=_get_env_float("SSDLITE_LR", 1e-3),
        optimizer=os.getenv("SSDLITE_OPTIMIZER", "Adam"),
        momentum=_get_env_float("SSDLITE_MOMENTUM", 0.9),
        weight_decay=_get_env_float("SSDLITE_WEIGHT_DECAY", 4e-5),
        lr_step_size=_get_env_int("SSDLITE_LR_STEP", 50),
        lr_gamma=_get_env_float("SSDLITE_LR_GAMMA", 0.5),
        num_workers=_get_env_int("SSDLITE_NUM_WORKERS", 4),
        score_thresh=_get_env_float("SSDLITE_SCORE_THRESH", 0.5),
        nms_thresh=_get_env_float("SSDLITE_NMS_THRESH", 0.5),
        detections_per_img=_get_env_int("SSDLITE_DETECTIONS_PER_IMG", 100),
        device=os.getenv("SSDLITE_DEVICE"),
    )


def get_training_params() -> dict:
    return asdict(get_config())


__all__ = ["SSDLiteConfig", "get_config", "get_training_params"]
