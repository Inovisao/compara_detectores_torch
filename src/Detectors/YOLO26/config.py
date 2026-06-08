from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "all" / "data_yolo26.yaml"
DEFAULT_WEIGHTS = os.getenv("YOLO26_WEIGHTS", "yolo26n.pt")
DEFAULT_PROJECT = SRC_ROOT / "runs" / "detect" / "YOLO26"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _env_int(name: str, default: str | int) -> int:
    return int(os.getenv(name, str(default)))


def get_training_params(data_yaml: str | Path | None = None) -> dict:
    data_path = Path(os.getenv("YOLO26_DATA", data_yaml or DEFAULT_DATA))

    return {
        "weights": DEFAULT_WEIGHTS,
        "data": str(data_path),
        "epochs": int(os.getenv("YOLO26_EPOCHS", "500")),
        "imgsz": int(os.getenv("YOLO26_IMGSZ", "640")),
        "patience": int(os.getenv("YOLO26_PATIENCE", "10")),
        "batch": int(os.getenv("YOLO26_BATCH", "16")),
        "project": os.getenv("YOLO26_PROJECT", str(DEFAULT_PROJECT)),
        "run_name": os.getenv("YOLO26_RUN_NAME", "train"),
        "optimizer": os.getenv("YOLO26_OPTIMIZER", "AdamW"),
        "single_cls": _env_bool("YOLO26_SINGLE_CLS", True),
        "rect": _env_bool("YOLO26_RECT", False),
        "cos_lr": _env_bool("YOLO26_COS_LR", True),
        "lr0": float(os.getenv("YOLO26_LR0", "0.001")),
        "lrf": float(os.getenv("YOLO26_LRF", "0.2")),
        "plots": _env_bool("YOLO26_PLOTS", True),
        "device": os.getenv("YOLO26_DEVICE"),
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
        plots=params["plots"],
        workers=params["workers"],
    )

    if params["device"]:
        train_kwargs["device"] = params["device"]

    print(model.model.loss)
    model.info(verbose=True)
    for name, param in model.model.named_parameters():
        if "dfl" in name.lower():
            print(name, param.shape)
    model.train(**train_kwargs)


def get_finetune_params(data_yaml: str | Path, weights: str | Path, fold: str = "") -> dict:
    data_path = Path(data_yaml)
    project   = SRC_ROOT / "runs" / "detect" / "YOLO26_finetune"
    suffix    = f"_{fold}" if fold else ""
    ft_patience = _env_int("YOLO26_FT_PATIENCE", 30)

    return {
        "phase_a": {
            "data":           str(data_path),
            "weights":        str(weights),
            "epochs":         int(os.getenv("YOLO26_FT_A_EPOCHS",    "300")),
            "imgsz":          int(os.getenv("YOLO26_FT_IMGSZ",       "640")),
            "patience":       _env_int("YOLO26_FT_A_PATIENCE", ft_patience),
            "batch":          int(os.getenv("YOLO26_FT_BATCH",        "32")),
            "freeze":         int(os.getenv("YOLO26_FT_A_FREEZE",     "0")),
            "lr0":          float(os.getenv("YOLO26_FT_A_LR0",     "0.01")),
            "lrf":          float(os.getenv("YOLO26_FT_A_LRF",      "0.01")),
            "warmup_epochs":  int(os.getenv("YOLO26_FT_A_WARMUP",     "3")),
            "optimizer":           os.getenv("YOLO26_FT_OPTIMIZER",   "MuSGD"),
            "momentum":     float(os.getenv("YOLO26_FT_MOMENTUM",    "0.937")),
            "weight_decay": float(os.getenv("YOLO26_FT_WD",        "0.0005")),
            "augment":      _env_bool("YOLO26_FT_TTA", False),
            "project":      str(project),
            "name":         f"phase_a{suffix}",
            "save_period":    int(os.getenv("YOLO26_FT_SAVE_PERIOD",  "5")),
             # augmentation explícito
            "hsv_h":    0.015,
            "hsv_s":    0.7,
            "hsv_v":    0.4,
            "degrees":  5.0,
            "translate": 0.1,
            "scale":    0.5,
            "fliplr":   0.5,
            "flipud":   0.0,
            "mosaic":   1.0,   # crítico para objetos pequenos
            "mixup":    0.1,
            "copy_paste": 0.1, # copia buracos pequenos para outras imagens
        },
        "phase_b": {
            "data":           str(data_path),
            "epochs":         int(os.getenv("YOLO26_FT_B_EPOCHS",    "50")),
            "imgsz":          int(os.getenv("YOLO26_FT_IMGSZ",       "640")),
            "patience":       _env_int("YOLO26_FT_B_PATIENCE", ft_patience),
            "batch":          int(os.getenv("YOLO26_FT_BATCH",        "16")),
            "freeze":         int(os.getenv("YOLO26_FT_B_FREEZE",     "9")),
            "lr0":          float(os.getenv("YOLO26_FT_B_LR0",    "0.0001")),
            "lrf":          float(os.getenv("YOLO26_FT_B_LRF",       "0.1")),
            "warmup_epochs":  int(os.getenv("YOLO26_FT_B_WARMUP",     "3")),
            "optimizer":           os.getenv("YOLO26_FT_OPTIMIZER",   "SGD"),
            "momentum":     float(os.getenv("YOLO26_FT_MOMENTUM",    "0.937")),
            "weight_decay": float(os.getenv("YOLO26_FT_WD",        "0.0005")),
            "augment":      _env_bool("YOLO26_FT_TTA", False),
            "project":      str(project),
            "name":         f"phase_b{suffix}",
            "save_period":    int(os.getenv("YOLO26_FT_SAVE_PERIOD",  "5")),
             # augmentation explícito
            "hsv_h":    0.015,
            "hsv_s":    0.7,
            "hsv_v":    0.4,
            "degrees":  5.0,
            "translate": 0.1,
            "scale":    0.5,
            "fliplr":   0.5,
            "flipud":   0.0,
            "mosaic":   1.0,   # crítico para objetos pequenos
            "mixup":    0.1,
            "copy_paste": 0.1, # copia buracos pequenos para outras imagens
        },
    }


def finetune(data_yaml: str | Path, weights: str | Path, fold: str = "") -> None:
    params = get_finetune_params(data_yaml, weights, fold)

    model = YOLO(str(weights))
    model.train(**{k: v for k, v in params["phase_a"].items() if k != "weights"})

    phase_a_best = (
        Path(params["phase_a"]["project"])
        / params["phase_a"]["name"]
        / "weights"
        / "best.pt"
    )
    model = YOLO(str(phase_a_best))
    model.train(**{k: v for k, v in params["phase_b"].items()})


if __name__ == "__main__":
    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
