from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data.yaml"
DEFAULT_WEIGHTS = os.getenv("YOLOV11_WEIGHTS", "yolo11s.pt")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def treino(data_yaml: str | Path | None = None) -> None:
    model = YOLO(DEFAULT_WEIGHTS)

    data_path = Path(os.getenv("YOLOV11_DATA", data_yaml or DEFAULT_DATA))
    epochs = int(os.getenv("YOLOV11_EPOCHS", "10"))
    imgsz = int(os.getenv("YOLOV11_IMGSZ", "640"))
    patience = int(os.getenv("YOLOV11_PATIENCE", "3"))
    batch = int(os.getenv("YOLOV11_BATCH", "8"))
    project = os.getenv("YOLOV11_PROJECT", "YOLOV11")
    run_name = os.getenv("YOLOV11_RUN_NAME", "train")
    optimizer = os.getenv("YOLOV11_OPTIMIZER", "AdamW")
    single_cls = _env_bool("YOLOV11_SINGLE_CLS", False)
    rect = _env_bool("YOLOV11_RECT", False)
    cos_lr = _env_bool("YOLOV11_COS_LR", True)
    lr0 = float(os.getenv("YOLOV11_LR0", "0.0005"))
    lrf = float(os.getenv("YOLOV11_LRF", "0.1"))
    plots = _env_bool("YOLOV11_PLOTS", True)
    device = os.getenv("YOLOV11_DEVICE")

    train_kwargs = dict(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        batch=batch,
        project=project,
        name=run_name,
        exist_ok=True,
        optimizer=optimizer,
        single_cls=single_cls,
        rect=rect,
        cos_lr=cos_lr,
        lr0=lr0,
        lrf=lrf,
        plots=plots,
    )

    if device:
        train_kwargs["device"] = device

    model.train(**train_kwargs)


if __name__ == "__main__":
    import sys

    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
