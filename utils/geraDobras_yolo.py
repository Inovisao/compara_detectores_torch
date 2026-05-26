from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Imagem nao encontrada ou invalida: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _collect_yolo_items(dataset_root: Path, source_dir: Path) -> tuple[list[dict], list[dict]]:
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    image_paths = sorted(
        path
        for path in source_dir.glob("*/images/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_id, image_path in enumerate(image_paths):
        width, height = _image_size(image_path)
        file_name = image_path.name
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": "",
            }
        )

        if not label_path.exists():
            continue

        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"Label invalida em {label_path}:{line_number}: {line}")

            class_id = int(parts[0])
            if class_id != 0:
                raise ValueError(
                    f"Classe inesperada em {label_path}:{line_number}: {class_id}. "
                    "Este script espera dataset binario com classe 0."
                )

            x_center, y_center, rel_width, rel_height = map(float, parts[1:])
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, rel_width, rel_height)):
                raise ValueError(f"Coordenada fora de [0, 1] em {label_path}:{line_number}: {line}")
            if rel_width <= 0.0 or rel_height <= 0.0:
                raise ValueError(f"Largura/altura invalida em {label_path}:{line_number}: {line}")

            box_width = rel_width * width
            box_height = rel_height * height
            x_min = (x_center * width) - (box_width / 2.0)
            y_min = (y_center * height) - (box_height / 2.0)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        round(max(0.0, x_min), 2),
                        round(max(0.0, y_min), 2),
                        round(min(box_width, width), 2),
                        round(min(box_height, height), 2),
                    ],
                    "area": round(box_width * box_height, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return images, annotations


def _annotations_for(images: Iterable[dict], annotations: list[dict]) -> list[dict]:
    image_ids = {int(image["id"]) for image in images}
    return [ann for ann in annotations if int(ann["image_id"]) in image_ids]


def _save_coco(path: Path, images: list[dict], annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {"description": "YOLO to COCO folds"},
        "licenses": [{"id": 1, "name": "unknown"}],
        "categories": [{"id": 1, "name": "pothole", "supercategory": "road_damage"}],
        "images": images,
        "annotations": annotations,
    }
    path.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_train_symlink(dataset_root: Path) -> None:
    train_link = dataset_root / "train"
    images_dir = dataset_root / "images"

    if train_link.exists() or train_link.is_symlink():
        return
    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada para criar symlink: {images_dir}")

    os.symlink(images_dir.resolve(), train_link)
    print(f"[symlink] {train_link} -> {images_dir}")


def generate_folds(dataset_root: Path, folds: int, valperc: float, seed: int) -> None:
    source_dir = dataset_root / "YOLO26"
    output_dir = dataset_root / "filesJSON"

    if not source_dir.exists():
        raise FileNotFoundError(f"Pasta YOLO nao encontrada: {source_dir}")

    images, annotations = _collect_yolo_items(dataset_root, source_dir)
    annotated_ids = {int(ann["image_id"]) for ann in annotations}
    images = [image for image in images if int(image["id"]) in annotated_ids]

    if not images:
        raise ValueError(f"Nenhuma imagem com anotacao encontrada em {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("fold_*.json"):
        old_file.unlink()

    test_size = len(images) // folds
    remaining = list(images)
    fold_splits: list[list[dict]] = []

    for fold_idx in range(folds - 1):
        remaining, test_fold = train_test_split(
            remaining,
            test_size=test_size,
            random_state=seed + fold_idx,
        )
        fold_splits.append(test_fold)
    fold_splits.append(remaining)

    print(f"[geraDobras_yolo] {len(images)} imagens | {len(annotations)} anotacoes | {folds} folds")

    for idx, test_images in enumerate(fold_splits, 1):
        train_val_images = [
            image
            for split_idx, split_images in enumerate(fold_splits, 1)
            if split_idx != idx
            for image in split_images
        ]
        train_images, val_images = train_test_split(
            train_val_images,
            test_size=valperc,
            random_state=seed + 100 + idx,
        )

        prefix = f"fold_{idx}"
        _save_coco(output_dir / f"{prefix}_train.json", train_images, _annotations_for(train_images, annotations))
        _save_coco(output_dir / f"{prefix}_val.json", val_images, _annotations_for(val_images, annotations))
        _save_coco(output_dir / f"{prefix}_test.json", test_images, _annotations_for(test_images, annotations))

        print(
            f"[fold {idx}] train={len(train_images)} "
            f"val={len(val_images)} test={len(test_images)}"
        )

    _ensure_train_symlink(dataset_root)
    print(f"[OK] JSONs salvos em {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dobras COCO a partir de um dataset YOLO.")
    parser.add_argument("--root", default="dataset/fine_tuning", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--valperc", default=0.3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_folds(
        dataset_root=args.root.resolve(),
        folds=args.folds,
        valperc=args.valperc,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
