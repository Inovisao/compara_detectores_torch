from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

from sklearn.model_selection import train_test_split


def _parse_xml(xml_path: Path) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    filename = root.findtext("filename") or (xml_path.stem + ".jpg")

    size = root.find("size")
    if size is None:
        return None
    width  = int(size.findtext("width",  "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return None

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if not name:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append({
            "name":      name,
            "xmin":      xmin,
            "ymin":      ymin,
            "xmax":      xmax,
            "ymax":      ymax,
            "truncated": int(obj.findtext("truncated", "0")),
            "difficult": int(obj.findtext("difficult",  "0")),
        })

    return {"filename": filename, "width": width, "height": height, "objects": objects}


def _build_coco(parsed: list[dict]) -> dict:
    all_names: set[str] = set()
    for p in parsed:
        for obj in p["objects"]:
            all_names.add(obj["name"])

    categories = [
        {"id": i + 1, "name": name, "supercategory": "road_damage"}
        for i, name in enumerate(sorted(all_names))
    ]
    name_to_id = {cat["name"]: cat["id"] for cat in categories}

    images, annotations = [], []
    ann_id = 1

    for img_id, p in enumerate(parsed):
        images.append({
            "id":            img_id,
            "file_name":     p["filename"],
            "width":         p["width"],
            "height":        p["height"],
            "license":       1,
            "date_captured": "",
        })
        for obj in p["objects"]:
            w = obj["xmax"] - obj["xmin"]
            h = obj["ymax"] - obj["ymin"]
            annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": name_to_id[obj["name"]],
                "bbox":        [round(obj["xmin"], 2), round(obj["ymin"], 2),
                                round(w, 2),            round(h, 2)],
                "area":        round(w * h, 2),
                "iscrowd":     0,
            })
            ann_id += 1

    return {
        "info":        {"description": "Pascal VOC to COCO"},
        "licenses":    [{"id": 1, "name": "unknown"}],
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }


def _filter_annotations(annotations: list[dict], images: list[dict]) -> list[dict]:
    ids = {img["id"] for img in images}
    return [a for a in annotations if a["image_id"] in ids]


def _save_coco(path: Path, coco: dict, images: list[dict], annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "info":        coco["info"],
                "licenses":    coco["licenses"],
                "categories":  coco["categories"],
                "images":      images,
                "annotations": annotations,
            },
            f, indent=2, ensure_ascii=False,
        )


def _apply_kfold(coco: dict, output_dir: Path, folds: int, valperc: float, having_annotations: bool) -> None:
    images      = coco["images"]
    annotations = coco["annotations"]

    if having_annotations:
        annotated_ids = {a["image_id"] for a in annotations}
        images = [img for img in images if img["id"] in annotated_ids]
        print(f"[kfold] {len(coco['images'])} → {len(images)} imagens (filtro: com anotação)")

    n        = len(images)
    qtd_test = n // folds
    print(f"[kfold] {n} imagens | {folds} folds | ~{qtd_test} imagens/teste")

    remaining    = list(images)
    fold_splits: list[list[dict]] = []
    for _ in range(folds - 1):
        remaining, test_fold = train_test_split(remaining, test_size=qtd_test)
        fold_splits.append(test_fold)
    fold_splits.append(remaining)

    for i in range(folds):
        print(f"[kfold] Fold {i + 1}/{folds} …")
        test_imgs     = fold_splits[i]
        train_val_imgs = [img for j, split in enumerate(fold_splits) if j != i for img in split]
        train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=valperc)

        prefix = f"fold_{i + 1}"
        _save_coco(output_dir / f"{prefix}_train.json", coco, train_imgs, _filter_annotations(annotations, train_imgs))
        _save_coco(output_dir / f"{prefix}_val.json",   coco, val_imgs,   _filter_annotations(annotations, val_imgs))
        _save_coco(output_dir / f"{prefix}_test.json",  coco, test_imgs,  _filter_annotations(annotations, test_imgs))
        print(f"  train={len(train_imgs)}  val={len(val_imgs)}  test={len(test_imgs)}")


def _ensure_train_symlink(root: Path) -> None:
    train_link = root / "train"
    images_dir = root / "images"
    if train_link.exists() or train_link.is_symlink():
        return
    if not images_dir.exists():
        print(f"[WARN] {images_dir} não encontrado — symlink train/ não criado")
        return
    os.symlink(images_dir.resolve(), train_link)
    print(f"[symlink] {train_link} → {images_dir}")


def main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent.parent
    root      = (project_root / args.root).resolve()
    xml_dir   = root / "annotations" / "xmls"
    output_dir = root / "filesJSON"

    if not xml_dir.exists():
        raise FileNotFoundError(f"Pasta de XMLs não encontrada: {xml_dir}")

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"[parse] {len(xml_files)} XMLs em {xml_dir}")

    parsed, skipped = [], 0
    for xml_path in xml_files:
        result = _parse_xml(xml_path)
        if result is None:
            skipped += 1
        else:
            parsed.append(result)

    annotated = sum(1 for p in parsed if p["objects"])
    print(f"[parse] {len(parsed)} válidos | {skipped} ignorados | {annotated} com objetos")

    coco    = _build_coco(parsed)
    classes = [c["name"] for c in coco["categories"]]
    print(f"[coco]  {len(coco['images'])} imagens | {len(coco['annotations'])} anotações | {len(classes)} classes: {classes}")

    _apply_kfold(coco, output_dir, args.folds, args.valperc, args.having_annotations)
    _ensure_train_symlink(root)
    print(f"\n[OK] {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="dataset/all", type=str)
    parser.add_argument("--folds",   default=5,   type=int)
    parser.add_argument("--valperc", default=0.3, type=float)
    parser.add_argument("--having-annotations", dest="having_annotations", action="store_true")
    main(parser.parse_args())
