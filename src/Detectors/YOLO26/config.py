from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data.yaml"
DEFAULT_FINETUNE_WEIGHTS = SRC_ROOT / "model_checkpoints" / "nano" / "fold_2" / "YOLO26" / "train" / "weights" / "best.pt"
DEFAULT_WEIGHTS = os.getenv("YOLO26_WEIGHTS", str(DEFAULT_FINETUNE_WEIGHTS))
DEFAULT_PROJECT = SRC_ROOT / "runs" / "detect" / "YOLO26"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalize_device(value: str | None) -> str:
    if value is None:
        return "gpu"

    normalized = value.strip().lower()
    if normalized in {"gpu", "cuda", "cuda:0", "0"}:
        return "gpu"
    if normalized == "cpu":
        return "cpu"
    return value


def _ultralytics_device(value: str) -> str:
    return "cuda:0" if value == "gpu" else value


def get_training_params(data_yaml: str | Path | None = None) -> dict:
    data_path = Path(os.getenv("YOLO26_DATA", data_yaml or DEFAULT_DATA))
    return {
        "weights": DEFAULT_WEIGHTS,
        "data": str(data_path),
        "epochs": int(os.getenv("YOLO26_EPOCHS", "100")),
        "imgsz": int(os.getenv("YOLO26_IMGSZ", "640")),
        "patience": int(os.getenv("YOLO26_PATIENCE", "20")),
        "batch": int(os.getenv("YOLO26_BATCH", "16")),
        "project": os.getenv("YOLO26_PROJECT", str(DEFAULT_PROJECT)),
        "run_name": os.getenv("YOLO26_RUN_NAME", "train"),
        "optimizer": os.getenv("YOLO26_OPTIMIZER", "SGD"),
        "single_cls": _env_bool("YOLO26_SINGLE_CLS", False),
        "rect": _env_bool("YOLO26_RECT", False),
        "cos_lr": _env_bool("YOLO26_COS_LR", True),
        "lr0": float(os.getenv("YOLO26_LR0", "0.0001")),
        "lrf": float(os.getenv("YOLO26_LRF", "0.01")),
        "weight_decay": float(os.getenv("YOLO26_WEIGHT_DECAY", "0.0005")),
        "freeze": os.getenv("YOLO26_FREEZE"),
        "plots": _env_bool("YOLO26_PLOTS", True),
        "device": _normalize_device(os.getenv("YOLO26_DEVICE")),
        "workers": int(os.getenv("YOLO26_WORKERS", "4")),
    }


def treino(data_yaml: str | Path | None = None) -> None:
    model = YOLO(DEFAULT_WEIGHTS)

    params = get_training_params(data_yaml)

    train_kwargs = dict(
        data=params["data"],
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        patience=params["patience"],
        batch=params["batch"],
        project=params["project"],
        name=params["run_name"],
        exist_ok=True,
        optimizer=params["optimizer"],
        single_cls=params["single_cls"],
        rect=params["rect"],
        cos_lr=params["cos_lr"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        weight_decay=params["weight_decay"],
        plots=params["plots"],
        workers=params["workers"],
    )

    if params["freeze"]:
        train_kwargs["freeze"] = int(params["freeze"])

    if params["device"]:
        train_kwargs["device"] = _ultralytics_device(params["device"])

    model.train(**train_kwargs)


if __name__ == "__main__":
    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
