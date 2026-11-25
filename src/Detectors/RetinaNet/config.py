from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from hyperparameters import param


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


def get_config() -> RetinaNetConfig:
    # Valores em src/hyperparameters.py (por dataset); RETINANET_* sobrescreve.
    p = lambda key, env: param("RetinaNet", key, f"RETINANET_{env}")
    return RetinaNetConfig(
        epochs=p("epochs", "EPOCHS"),
        batch_size=p("batch_size", "BATCH"),
        learning_rate=p("learning_rate", "LR"),
        momentum=p("momentum", "MOMENTUM"),
        weight_decay=p("weight_decay", "WEIGHT_DECAY"),
        lr_step_size=p("lr_step_size", "LR_STEP"),
        lr_gamma=p("lr_gamma", "LR_GAMMA"),
        num_workers=p("num_workers", "NUM_WORKERS"),
        device=p("device", "DEVICE"),
    )


def get_training_params() -> dict:
    return asdict(get_config())


__all__ = ["RetinaNetConfig", "get_config", "get_training_params"]
