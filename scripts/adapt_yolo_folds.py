#!/usr/bin/env python3
"""
Reorganiza o dataset SAHI para o formato esperado por main.py.

Estrutura atual (por fold, isolada):
  dataset/all/
    fold_N/
      filesJSON/fold_N_{train,val,test}.json   ← COCO já pronto
      train/images/   ← tiles 640×640 + FI
      val/images/     ← imagens originais
      test/images/    ← imagens originais

main.py espera:
  dataset/all/
    filesJSON/fold_N_{train,val,test}.json     ← link para os JSONs de cada fold
    train/                                     ← todas as imagens únicas (hardlink)
    train/_annotations.coco.json              ← COCO completo (lido pelo config.py do DETR)

Uso:
  python scripts/adapt_yolo_folds.py
  python scripts/adapt_yolo_folds.py --dry-run
  python scripts/adapt_yolo_folds.py --dataset /outro/caminho/all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _discover_folds(dataset_root: Path) -> list[int]:
    fold_nums = sorted(
        int(p.name.split("_")[1])
        for p in dataset_root.iterdir()
        if p.is_dir() and p.name.startswith("fold_") and p.name.count("_") == 1
    )
    if not fold_nums:
        raise FileNotFoundError(f"Nenhuma pasta fold_N encontrada em {dataset_root}")
    return fold_nums


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_or_copy_images(images_dir: Path, train_flat: Path, dry_run: bool) -> int:
    """Hardlinka (ou copia) imagens de images_dir para train_flat. Retorna qtd copiada."""
    if not images_dir.exists():
        return 0
    copied = 0
    for img in images_dir.iterdir():
        if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
            dst = train_flat / img.name
            if not dst.exists():
                if not dry_run:
                    _link_or_copy_file(img, dst)
                copied += 1
    return copied


def _build_combined_coco(json_paths: list[Path]) -> dict:
    """Merge múltiplos COCO JSONs em um único, reindexando IDs para evitar colisões."""
    combined: dict = {
        "info": {"description": "Combined COCO for DETR class discovery"},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    categories_set: dict[int, dict] = {}
    next_img_id = 1
    next_ann_id = 1

    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))

        for cat in data.get("categories", []):
            categories_set[cat["id"]] = cat

        img_id_remap: dict[int, int] = {}
        for img in data.get("images", []):
            new_id = next_img_id
            img_id_remap[img["id"]] = new_id
            combined["images"].append({**img, "id": new_id})
            next_img_id += 1

        for ann in data.get("annotations", []):
            combined["annotations"].append({
                **ann,
                "id": next_ann_id,
                "image_id": img_id_remap[ann["image_id"]],
            })
            next_ann_id += 1

    combined["categories"] = list(categories_set.values())
    return combined


def adapt(dataset_root: Path, dry_run: bool) -> None:
    folds = _discover_folds(dataset_root)
    print(f"[INFO] Folds encontrados : {folds}")

    files_json_dir = dataset_root / "filesJSON"
    train_flat_dir = dataset_root / "train"

    if not dry_run:
        files_json_dir.mkdir(exist_ok=True)
        train_flat_dir.mkdir(exist_ok=True)

    train_json_paths: list[Path] = []
    total_imgs_linked = 0
    total_jsons_linked = 0

    for fold_n in folds:
        fold_dir = dataset_root / f"fold_{fold_n}"
        src_json_dir = fold_dir / "filesJSON"

        for split in SPLITS:
            src_json = src_json_dir / f"fold_{fold_n}_{split}.json"
            dst_json = files_json_dir / f"fold_{fold_n}_{split}.json"

            if not src_json.exists():
                print(f"[SKIP] JSON não encontrado: {src_json}")
                continue

            if not dry_run:
                if not dst_json.exists():
                    _link_or_copy_file(src_json, dst_json)
            total_jsons_linked += 1

            images_dir = fold_dir / split / "images"
            n = _link_or_copy_images(images_dir, train_flat_dir, dry_run)
            total_imgs_linked += n

            if split == "train":
                train_json_paths.append(src_json)

            print(f"  [fold_{fold_n}/{split}] JSON ✓  |  {n} imagens novas → train/")

    # _annotations.coco.json para o config.py do DETR
    annotations_path = train_flat_dir / "_annotations.coco.json"
    if not dry_run and not annotations_path.exists():
        combined = _build_combined_coco(train_json_paths)
        annotations_path.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        cats = [c["name"] for c in combined["categories"]]
        print(f"\n[OK] _annotations.coco.json → {len(combined['images'])} imgs | classes: {cats}")

    print(f"\n[OK] filesJSON/  → {total_jsons_linked} JSONs")
    print(f"[OK] train/      → {total_imgs_linked} imagens novas linkadas")

    if dry_run:
        print("\n[DRY-RUN] Nenhum arquivo foi criado ou modificado.")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Reorganiza dataset SAHI para o formato esperado por main.py."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset" / "all",
        help="Raiz do dataset (contém fold_1/, fold_2/, …). Padrão: dataset/all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria feito sem criar arquivos.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root não encontrado: {dataset_root}")
    adapt(dataset_root=dataset_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
