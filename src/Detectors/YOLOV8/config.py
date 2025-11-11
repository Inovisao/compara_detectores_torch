from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO

# https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treino

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data.yaml"
DEFAULT_WEIGHTS = os.getenv("YOLOV8_WEIGHTS", "yolov8s.pt")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def treino(data_yaml: str | Path | None = None) -> None:
    model = YOLO(DEFAULT_WEIGHTS)

    data_path = Path(os.getenv("YOLOV8_DATA", data_yaml or DEFAULT_DATA))
    epochs = int(os.getenv("YOLOV8_EPOCHS", "500"))
    imgsz = int(os.getenv("YOLOV8_IMGSZ", "640"))
    patience = int(os.getenv("YOLOV8_PATIENCE", "50"))
    batch = int(os.getenv("YOLOV8_BATCH", "4"))
    project = os.getenv("YOLOV8_PROJECT", "YOLOV8")
    run_name = os.getenv("YOLOV8_RUN_NAME", "train")
    optimizer = os.getenv("YOLOV8_OPTIMIZER", "SGD")
    single_cls = _env_bool("YOLOV8_SINGLE_CLS", False)
    rect = _env_bool("YOLOV8_RECT", False)
    cos_lr = _env_bool("YOLOV8_COS_LR", True)
    lr0 = float(os.getenv("YOLOV8_LR0", "0.001"))
    lrf = float(os.getenv("YOLOV8_LRF", "0.1"))
    plots = _env_bool("YOLOV8_PLOTS", True)
    device = os.getenv("YOLOV8_DEVICE")
    workers = int(os.getenv("YOLOV8_WORKERS", "4"))

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
        workers=workers,
    )

    if device:
        train_kwargs["device"] = device

    model.train(**train_kwargs)


if __name__ == "__main__":
    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
