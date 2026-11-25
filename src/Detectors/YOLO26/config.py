from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO

from hyperparameters import param


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data.yaml"
DEFAULT_PROJECT = SRC_ROOT / "runs" / "detect" / "YOLO26"


def _resolve_weights(weights: str) -> str:
    """Pesos relativos a src/ viram caminho absoluto quando o arquivo existe."""
    candidate = SRC_ROOT / weights
    return str(candidate) if not Path(weights).is_absolute() and candidate.exists() else weights


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
    # Valores em src/hyperparameters.py (por dataset); YOLO26_* sobrescreve.
    data_path = Path(os.getenv("YOLO26_DATA", data_yaml or DEFAULT_DATA))
    p = lambda key: param("YOLO26", key, f"YOLO26_{key.upper()}")
    return {
        "weights": _resolve_weights(p("weights")),
        "data": str(data_path),
        "epochs": p("epochs"),
        "imgsz": p("imgsz"),
        "patience": p("patience"),
        "batch": p("batch"),
        "project": os.getenv("YOLO26_PROJECT", str(DEFAULT_PROJECT)),
        "run_name": os.getenv("YOLO26_RUN_NAME", "train"),
        "optimizer": p("optimizer"),
        "single_cls": p("single_cls"),
        "rect": p("rect"),
        "cos_lr": p("cos_lr"),
        "lr0": p("lr0"),
        "lrf": p("lrf"),
        "weight_decay": p("weight_decay"),
        "freeze": p("freeze"),
        "plots": p("plots"),
        "device": _normalize_device(p("device")),
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
