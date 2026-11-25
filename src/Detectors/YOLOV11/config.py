from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO

from hyperparameters import param


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data.yaml"
DEFAULT_PROJECT = SRC_ROOT / "runs" / "detect" / "YOLOV11"


def get_training_params(data_yaml: str | Path | None = None) -> dict:
    # Valores em src/hyperparameters.py (por dataset); YOLOV11_* sobrescreve.
    data_path = Path(os.getenv("YOLOV11_DATA", data_yaml or DEFAULT_DATA))
    p = lambda key: param("YOLOV11", key, f"YOLOV11_{key.upper()}")
    return {
        "weights": p("weights"),
        "data": str(data_path),
        "epochs": p("epochs"),
        "imgsz": p("imgsz"),
        "patience": p("patience"),
        "batch": p("batch"),
        "project": os.getenv("YOLOV11_PROJECT", str(DEFAULT_PROJECT)),
        "run_name": os.getenv("YOLOV11_RUN_NAME", "train"),
        "optimizer": p("optimizer"),
        "single_cls": p("single_cls"),
        "rect": p("rect"),
        "cos_lr": p("cos_lr"),
        "lr0": p("lr0"),
        "lrf": p("lrf"),
        "weight_decay": p("weight_decay"),
        "plots": p("plots"),
        "device": p("device"),
        "workers": p("workers"),
    }


def treino(data_yaml: str | Path | None = None) -> None:
    params = get_training_params(data_yaml)
    model = YOLO(params["weights"])

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
        workers=params["workers"],
        plots=params["plots"],
    )

    if params["device"]:
        train_kwargs["device"] = params["device"]

    model.train(**train_kwargs)


if __name__ == "__main__":
    import sys

    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
