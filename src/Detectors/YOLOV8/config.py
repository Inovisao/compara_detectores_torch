from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO

# https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treino

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data.yaml"
DEFAULT_WEIGHTS = os.getenv("YOLOV8_WEIGHTS", "yolov8s.pt")
DEFAULT_PROJECT = SRC_ROOT / "runs" / "detect" / "YOLOV8"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def get_training_params(data_yaml: str | Path | None = None) -> dict:
    data_path = Path(os.getenv("YOLOV8_DATA", data_yaml or DEFAULT_DATA))
    return {
        "weights": DEFAULT_WEIGHTS,
        "data": str(data_path),
        "epochs": int(os.getenv("YOLOV8_EPOCHS", "1000")),
        "imgsz": int(os.getenv("YOLOV8_IMGSZ", "640")),
        "patience": int(os.getenv("YOLOV8_PATIENCE", "50")),
        "batch": int(os.getenv("YOLOV8_BATCH", "32")),
        "project": os.getenv("YOLOV8_PROJECT", str(DEFAULT_PROJECT)),
        "run_name": os.getenv("YOLOV8_RUN_NAME", "train"),
        "optimizer": os.getenv("YOLOV8_OPTIMIZER", "SGD"),
        "single_cls": _env_bool("YOLOV8_SINGLE_CLS", True),
        "rect": _env_bool("YOLOV8_RECT", False),
        "cos_lr": _env_bool("YOLOV8_COS_LR", True),
        "lr0": float(os.getenv("YOLOV8_LR0", "0.01")),
        "lrf": float(os.getenv("YOLOV8_LRF", "0.2")),
        "momentum": float(os.getenv("YOLOV8_MOMENTUM", "0.937")),
        "weight_decay": float(os.getenv("YOLOV8_WEIGHT_DECAY", "0.0005")),
        "mosaic": float(os.getenv("YOLOV8_MOSAIC", "0.0")),
        "mixup": float(os.getenv("YOLOV8_MIXUP", "0.0")),
        "copy_paste": float(os.getenv("YOLOV8_COPY_PASTE", "0.0")),
        "fliplr": float(os.getenv("YOLOV8_FLIPLR", "0.0")),
        "flipud": float(os.getenv("YOLOV8_FLIPUD", "0.0")),
        "hsv_h": float(os.getenv("YOLOV8_HSV_H", "0.0")),
        "hsv_s": float(os.getenv("YOLOV8_HSV_S", "0.0")),
        "hsv_v": float(os.getenv("YOLOV8_HSV_V", "0.0")),
        "translate": float(os.getenv("YOLOV8_TRANSLATE", "0.0")),
        "scale": float(os.getenv("YOLOV8_SCALE", "0.0")),
        "degrees": float(os.getenv("YOLOV8_DEGREES", "0.0")),
        "plots": _env_bool("YOLOV8_PLOTS", True),
        "device": os.getenv("YOLOV8_DEVICE"),
        "workers": int(os.getenv("YOLOV8_WORKERS", "8")),
    }


def treino(data_yaml: str | Path | None = None) -> None:
    try:
        import ultralytics.data.augment as _aug
        class _NoopAlbumentations:
            def __init__(self, *a, **kw): pass
            def __call__(self, labels): return labels
        _aug.Albumentations = _NoopAlbumentations
    except Exception:
        pass

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
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
        mosaic=params["mosaic"],
        mixup=params["mixup"],
        copy_paste=params["copy_paste"],
        fliplr=params["fliplr"],
        flipud=params["flipud"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        translate=params["translate"],
        scale=params["scale"],
        degrees=params["degrees"],
        plots=params["plots"],
        workers=params["workers"],
    )

    if params["device"]:
        train_kwargs["device"] = params["device"]

    model.train(**train_kwargs)


if __name__ == "__main__":
    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
