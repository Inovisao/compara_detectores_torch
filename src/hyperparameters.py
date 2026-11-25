"""Hiperparâmetros de todos os detectores, separados por dataset (first / second).

Edite os valores AQUI. Cada `config.py` de detector lê esta tabela.

Ordem de prioridade para cada parâmetro:
  1. variável de ambiente do detector (ex.: YOLO26_LR0=0.0005) — sobrescrita pontual;
  2. valor de HYPERPARAMETERS[<dataset>][<detector>][<chave>];

O dataset vem de DATASET_NAME / DATASET_ROOT (ver result_paths.py). Se o nome
do dataset não estiver na tabela, usa-se DATASET_FALLBACK.

Fluxo do README: o `first` treina a partir dos pesos base; o `second` faz
fine-tuning a partir do melhor checkpoint do `first`.

ATENÇÃO: os valores do `first` do YOLO26 são um ponto de partida (iguais aos
das demais YOLO) — ajuste-os. Nos demais detectores, `first` e `second` partem
dos mesmos valores que já eram usados antes desta tabela.
"""
from __future__ import annotations

import os
import sys

from result_paths import DATASET_NAME

DATASET_FALLBACK = "second"

# Pesos do fine-tuning (relativo a src/): melhor checkpoint do `first`.
_YOLO26_FIRST_BEST = "model_checkpoints/nano/fold_2/YOLO26/train/weights/best.pt"

_FIRST = {
    "YOLO26": {
        "weights": "yolo26n.pt",
        "epochs": 1000, "imgsz": 640, "patience": 100, "batch": 16,
        "optimizer": "SGD", "single_cls": False, "rect": False, "cos_lr": True,
        "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005,
        "freeze": None, "plots": True, "device": None, "workers": 4,
    },
    "YOLOV8": {
        "weights": "yolov8s.pt",
        "epochs": 1000, "imgsz": 640, "patience": 100, "batch": 16,
        "optimizer": "SGD", "single_cls": False, "rect": False, "cos_lr": True,
        "lr0": 0.001, "lrf": 0.2, "weight_decay": 0.0005,
        "plots": True, "device": None, "workers": 4,
    },
    "YOLOV11": {
        "weights": "yolo11s.pt",
        "epochs": 1000, "imgsz": 640, "patience": 100, "batch": 16,
        "optimizer": "SGD", "single_cls": False, "rect": False, "cos_lr": True,
        "lr0": 0.001, "lrf": 0.2, "weight_decay": 0.0005,
        "plots": True, "device": None, "workers": 8,
    },
    "YOLOV5_TPH": {
        "cfg": "yolov5s.yaml", "img_size": 640,
        "epochs": 1000, "patience": 150, "batch": 8,
        "optimizer": "SGD", "single_cls": False, "rect": False, "cos_lr": True,
        "lr0": 0.001, "lrf": 0.1, "plots": True,
        "hyp_path": None, "pretrained_weights": None, "device": None,
    },
    "RetinaNet": {
        "epochs": 1000, "batch_size": 4, "learning_rate": 1e-4,
        "momentum": 0.9, "weight_decay": 1e-4,
        "lr_step_size": 5, "lr_gamma": 0.1, "num_workers": 4, "device": None,
    },
    "Faster": {
        "epochs": 1000, "batch_size": 4, "resize_to": 640, "num_workers": 4,
        "learning_rate": 1e-4, "momentum": 0.9, "weight_decay": 5e-4,
        "lr_step_size": 3, "lr_gamma": 0.1, "patience": 150,
    },
    "Detr": {
        "epochs": 1000, "batch_size": 8, "resize_to": 640, "num_workers": 4,
        # valores que o train_detector.py de fato usava (o LR=1e-4 do config antigo era ignorado)
        "learning_rate": 5e-5, "lr_backbone": 1e-6, "weight_decay": 1e-4,
    },
}

# `second`: começa igual ao `first` e sobrescreve só o que muda no fine-tuning.
# Fine-tuning curto e com lr menor, partindo do melhor checkpoint do `first`.
_SECOND = {name: dict(values) for name, values in _FIRST.items()}
_SECOND["YOLO26"].update(
    weights=_YOLO26_FIRST_BEST,
    epochs=100, patience=20, lr0=0.0001,
)

HYPERPARAMETERS = {
    "first": _FIRST,
    "second": _SECOND,
}


def _table() -> dict:
    if DATASET_NAME in HYPERPARAMETERS:
        return HYPERPARAMETERS[DATASET_NAME]
    print(
        f"[AVISO] Dataset '{DATASET_NAME}' não está em hyperparameters.py; "
        f"usando os valores de '{DATASET_FALLBACK}'.",
        file=sys.stderr,
    )
    return HYPERPARAMETERS[DATASET_FALLBACK]


def param(detector: str, key: str, env_name: str | None = None):
    """Valor de `key` para `detector` no dataset atual, com override por env."""
    default = _table()[detector][key]
    value = os.getenv(env_name) if env_name else None
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value  # str ou None (ex.: device, freeze, pretrained_weights)

