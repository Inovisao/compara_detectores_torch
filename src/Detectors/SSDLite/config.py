from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from losses import loss_weights_from_env
from losses import normalize_box_loss


@dataclass(frozen=True)
class SSDLiteConfig:
    backbone: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    momentum: float
    weight_decay: float
    lr_step_size: int
    lr_gamma: float
    num_workers: int
    pin_memory: bool
    score_thresh: float
    nms_thresh: float
    detections_per_img: int
    device: str | None
    patience: int
    loss_weights: dict[str, float]
    box_loss: str
    inner_ratio: float


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_config() -> SSDLiteConfig:
    return SSDLiteConfig(
        backbone=os.getenv("SSDLITE_BACKBONE", "mobilenetv2"),
        epochs=_get_env_int("SSDLITE_EPOCHS", 80),
        batch_size=_get_env_int("SSDLITE_BATCH", 8),
        learning_rate=_get_env_float("SSDLITE_LR", 5e-3),
        optimizer=os.getenv("SSDLITE_OPTIMIZER", "SGD"),
        momentum=_get_env_float("SSDLITE_MOMENTUM", 0.9),
        weight_decay=_get_env_float("SSDLITE_WEIGHT_DECAY", 4e-5),
        lr_step_size=_get_env_int("SSDLITE_LR_STEP", 50),
        lr_gamma=_get_env_float("SSDLITE_LR_GAMMA", 0.5),
        num_workers=_get_env_int("SSDLITE_NUM_WORKERS", 0),
        pin_memory=_get_env_bool("SSDLITE_PIN_MEMORY", False),
        score_thresh=_get_env_float("SSDLITE_SCORE_THRESH", 0.01),
        nms_thresh=_get_env_float("SSDLITE_NMS_THRESH", 0.45),
        detections_per_img=_get_env_int("SSDLITE_DETECTIONS_PER_IMG", 200),
        device=os.getenv("SSDLITE_DEVICE"),
        patience=_get_env_int("SSDLITE_PATIENCE", 15),
        loss_weights=loss_weights_from_env(
            "SSDLITE",
            {
                "classification": 1.0,
                "bbox_regression": 1.0,
            },
        ),
        box_loss=normalize_box_loss(os.getenv("SSDLITE_BOX_LOSS", "ciou")),
        inner_ratio=_get_env_float("SSDLITE_INNER_RATIO", 0.7),
    )


def get_training_params() -> dict:
    return asdict(get_config())


__all__ = ["SSDLiteConfig", "get_config", "get_training_params"]
