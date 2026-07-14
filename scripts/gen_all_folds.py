#!/usr/bin/env python3
"""
Gera filesJSON/ para o dataset 'all' (imagens originais sem tiles).

Estratégia: deriva os splits test/val dos mesmos filesJSON do asahi_rect
(que foi gerado com seed=42), garantindo que as mesmas imagens originais
estejam nos mesmos folds em todos os datasets (all, asahi, asahi_rect, sahi).

Além de criar filesJSON/, cria as pastas train/, val/, test/ com hardlinks
para todas as imagens (necessário para GeraLabels.py e GeraDobras.py).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

SLICE_DS = Path("/home/neto/development/slice_inference_api/dataset")
ALL_DIR   = SLICE_DS / "all"
RECT_DIR  = SLICE_DS / "asahi_rect"
ANN_FILE  = ALL_DIR / "_annotations.coco.json"
FILES_DIR = ALL_DIR / "filesJSON"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _make_split(
    split_filenames: set[str],
    all_imgs_by_name: dict,
    all_anns_by_img_id: dict,
    categories: list,
    info: dict,
    licenses: list,
) -> dict:
    images_out = []
    annotations_out = []
    new_img_id = 1
    new_ann_id = 1
    old_to_new_id: dict[int, int] = {}

    for fname in sorted(split_filenames):
        img = all_imgs_by_name.get(fname)
        if img is None:
            print(f"[warn] {fname} not found in all/_annotations.coco.json, skipping")
            continue
        old_id = img["id"]
        old_to_new_id[old_id] = new_img_id
        new_img = dict(img)
        new_img["id"] = new_img_id
        images_out.append(new_img)
        new_img_id += 1

    for old_img_id, new_img_id_val in old_to_new_id.items():
        for ann in all_anns_by_img_id.get(old_img_id, []):
            new_ann = dict(ann)
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = new_img_id_val
            annotations_out.append(new_ann)
            new_ann_id += 1

    return {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images_out,
        "annotations": annotations_out,
    }


def main() -> None:
    print(f"[gen_all_folds] Loading {ANN_FILE}")
    data = _load_json(ANN_FILE)

    all_imgs_by_name = {img["file_name"]: img for img in data["images"]}
    all_anns_by_img_id: dict[int, list] = {}
    for ann in data["annotations"]:
        all_anns_by_img_id.setdefault(ann["image_id"], []).append(ann)

    all_filenames = set(all_imgs_by_name.keys())
    categories = data["categories"]
    info = data.get("info", {})
    licenses = data.get("licenses", [])

    FILES_DIR.mkdir(exist_ok=True)

    for fold_num in range(1, 6):
        fold = f"fold_{fold_num}"

        test_json = _load_json(RECT_DIR / "filesJSON" / f"{fold}_test.json")
        val_json  = _load_json(RECT_DIR / "filesJSON" / f"{fold}_val.json")

        test_names = {img["file_name"] for img in test_json["images"]}
        val_names  = {img["file_name"] for img in val_json["images"]}
        train_names = all_filenames - test_names - val_names

        for split_name, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
            split_data = _make_split(
                names, all_imgs_by_name, all_anns_by_img_id, categories, info, licenses
            )
            out_path = FILES_DIR / f"{fold}_{split_name}.json"
            _save_json(out_path, split_data)
            print(f"[gen_all_folds] {fold}_{split_name}: {len(split_data['images'])} images, "
                  f"{len(split_data['annotations'])} annotations")

    # ── Create train/, val/, test/ dirs with hardlinks to all images ─────────
    # All images go to all dirs so GeraLabels.py/GeraDobras.py can always find them.
    print("[gen_all_folds] Creating train/, val/, test/ dirs with hardlinks...")
    for split in ("train", "val", "test"):
        (ALL_DIR / split).mkdir(exist_ok=True)

    for fname, img in all_imgs_by_name.items():
        src = ALL_DIR / fname
        if not src.exists():
            print(f"[warn] image not found: {src}")
            continue
        for split in ("train", "val", "test"):
            dst = ALL_DIR / split / fname
            if not dst.exists():
                os.link(src, dst)

    # ── Copy _annotations.coco.json to train/ (needed by DETR config) ────────
    ann_dst = ALL_DIR / "train" / "_annotations.coco.json"
    if not ann_dst.exists():
        shutil.copy(ANN_FILE, ann_dst)
        print(f"[gen_all_folds] Copied _annotations.coco.json to train/")

    print("[gen_all_folds] Done.")


if __name__ == "__main__":
    main()
