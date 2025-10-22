#!/usr/bin/env python3
"""Avalia um checkpoint específico usando o pipeline existente."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _parse_args() -> argparse.Namespace:
    default_model = PROJECT_ROOT / "src" / "model_checkpoints" / "fold_1" / "YOLOV5_TPH" / "train" / "weights" / "best.pt"
    default_dataset = PROJECT_ROOT / "dataset" / "tiles" / "sage" / "fold_1"

    parser = argparse.ArgumentParser(
        description="Executa a avaliação de um checkpoint .pt/.pth usando o pipeline principal."
    )
    parser.add_argument(
        "--model",
        default=str(default_model),
        help="Caminho para o arquivo de pesos (ex.: src/model_checkpoints/fold_1/YOLOV5_TPH/train/weights/best.pt).",
    )
    parser.add_argument(
        "--model-name",
        default="YOLOV5_TPH",
        help="Nome do modelo (ex.: YOLOV8, YOLOV11, YOLOV5_TPH, Faster, RetinaNet, Detr).",
    )
    parser.add_argument(
        "--fold",
        default="fold_1",
        help="Identificador da dobra (ex.: fold_1).",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset),
        help="Diretório raiz da dobra (ex.: dataset/tiles/sage/fold_1).",
    )
    parser.add_argument(
        "--save-imgs",
        default="../results/prediction/fold_1",
        action="store_true",
        help="Se definido, imagens anotadas serão salvas em results/prediction.",
    )
    parser.add_argument(
        "--tiling-mode",
        default="auto",
        choices=["auto", "sage", "basic", "normal", "none"],
        help="Força o modo de tiling a ser usado na agregação (auto detecta automaticamente).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    model_path = Path(args.model).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Diretório da dobra não encontrado: {dataset_root}")

    os.chdir(SRC_DIR)

    from ResultsDetections import create_csv  # noqa: WPS433

    print(f"[INFO] Avaliando {model_path} no dataset {dataset_root} (modelo={args.model_name})")
    create_csv(
        root=str(dataset_root),
        fold=args.fold,
        selected_model=args.model_name,
        model_path=str(model_path),
        save_imgs=args.save_imgs,
        tiling_mode=args.tiling_mode,
    )
    print("[INFO] Avaliação concluída")


if __name__ == "__main__":
    main()
