#!/usr/bin/env python3
"""
Gera filesJSON/ por fold convertendo as labels YOLO (.txt) de cada split
para COCO JSON — sem usar _annotations.coco.json como fonte de coordenadas.

As labels YOLO já estão em coordenadas normalizadas relativas a cada imagem
(tile ou imagem original), então a conversão é direta:
  x_min = (cx - w/2) * img_width
  y_min = (cy - h/2) * img_height
  bbox_w = w * img_width
  bbox_h = h * img_height

Uso:
    python scripts/gen_fold_jsons.py
    python scripts/gen_fold_jsons.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS     = ["sahi", "asahi", "asahi_rect"]
CATEGORIES   = [{"id": 1, "name": "insect", "supercategory": "insect"}]
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _read_image_size(path: Path) -> tuple[int, int]:
    """Retorna (width, height) sem depender de cv2."""
    with open(path, "rb") as f:
        header = f.read(12)

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        with open(path, "rb") as f:
            f.seek(16)
            w = struct.unpack(">I", f.read(4))[0]
            h = struct.unpack(">I", f.read(4))[0]
        return w, h

    # JPEG
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        pass

    # fallback: parse JPEG markers manually
    with open(path, "rb") as f:
        f.read(2)
        while True:
            marker = f.read(2)
            if len(marker) < 2:
                break
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if marker in (b"\xff\xc0", b"\xff\xc1", b"\xff\xc2"):
                f.read(1)
                h = struct.unpack(">H", f.read(2))[0]
                w = struct.unpack(">H", f.read(2))[0]
                return w, h
            f.seek(length - 2, 1)
    raise ValueError(f"Could not read image size: {path}")


def _yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float,
                        img_w: int, img_h: int) -> list[float]:
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    bbox_w = w * img_w
    bbox_h = h * img_h
    # clamp negatives from floating-point edge effects
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    return [round(x_min, 4), round(y_min, 4), round(bbox_w, 4), round(bbox_h, 4)]


def _build_coco_from_yolo(
    images_dir: Path,
    labels_dir: Path,
    dry_run: bool = False,
) -> dict:
    images, annotations = [], []
    img_id, ann_id = 1, 1

    img_files = sorted(
        p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTS
    )

    for img_path in img_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        if dry_run:
            img_w, img_h = 640, 640  # fast placeholder
        else:
            img_w, img_h = _read_image_size(img_path)

        img_entry = {
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        }
        images.append(img_entry)

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            bbox = _yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,  # labels use COCO category ids directly
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 4),
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return {
        "info": {},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def process_dataset(ds_path: Path, dry_run: bool) -> None:
    print(f"\n=== {ds_path.name} ===")
    for fold_n in range(1, 6):
        fold_dir = ds_path / f"fold_{fold_n}"
        if not fold_dir.exists():
            print(f"  [SKIP] fold_{fold_n}: pasta não encontrada")
            continue

        fj_dir = fold_dir / "filesJSON"
        if not dry_run:
            fj_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            images_dir = fold_dir / split / "images"
            labels_dir = fold_dir / split / "labels"
            out_path   = fj_dir / f"fold_{fold_n}_{split}.json"

            if not images_dir.exists() or not labels_dir.exists():
                print(f"  [SKIP] fold_{fold_n}/{split}: imagens ou labels não encontrados")
                continue

            built = _build_coco_from_yolo(images_dir, labels_dir, dry_run=dry_run)
            n_imgs = len(built["images"])
            n_anns = len(built["annotations"])

            if not dry_run:
                with open(out_path, "w") as f:
                    json.dump(built, f)

            tag = "[dry]" if dry_run else "[ok]"
            print(f"  {tag} fold_{fold_n}/{split}: {n_imgs} imgs, {n_anns} anns → {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ds_root = PROJECT_ROOT / "dataset"
    for ds_name in DATASETS:
        ds_path = ds_root / ds_name
        if not ds_path.exists():
            print(f"\n[SKIP] {ds_name}: pasta não encontrada")
            continue
        process_dataset(ds_path, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] Nenhum arquivo criado.")


if __name__ == "__main__":
    main()
