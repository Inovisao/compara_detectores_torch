from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from Detectors.FasterRCNN.geradataset import FasterDatasetConfig, geredata
from utils.augmentation import build_augmentation_pipeline

_PIPELINE = build_augmentation_pipeline("coco")


def _augment_coco_train(
    original_json_path: Path,
    images_src: Path,
    output_images_dir: Path,
    output_json_path: Path,
    copies: int,
) -> None:
    with open(original_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_images_dir.mkdir(parents=True, exist_ok=True)

    ann_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    next_image_id = max((img["id"] for img in data["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in data["annotations"]), default=0) + 1

    new_images = list(data["images"])
    new_annotations = list(data["annotations"])

    for image_info in data["images"]:
        src_path = images_src / image_info["file_name"]
        shutil.copy(src_path, output_images_dir / image_info["file_name"])

        image_bgr = cv2.imread(str(src_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        anns = ann_by_image.get(image_info["id"], [])
        bboxes = [ann["bbox"] for ann in anns]
        class_labels = [ann["category_id"] for ann in anns]

        stem, ext = image_info["file_name"].rsplit(".", 1)

        for i in range(copies):
            result = _PIPELINE(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            aug_file_name = f"{stem}_aug{i + 1}.{ext}"
            aug_h, aug_w = result["image"].shape[:2]

            aug_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_images_dir / aug_file_name), aug_bgr)

            new_images.append({
                **image_info,
                "id": next_image_id,
                "file_name": aug_file_name,
                "width": aug_w,
                "height": aug_h,
            })

            for aug_bbox, cat_id in zip(result["bboxes"], result["class_labels"]):
                x, y, w, h = [round(v, 2) for v in aug_bbox]
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                next_ann_id += 1

            next_image_id += 1

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({**data, "images": new_images, "annotations": new_annotations}, f, ensure_ascii=False)


def GeraDobrasRCNN(fold: str, root_dir: str | Path, augmentation_copies: int = 2) -> FasterDatasetConfig:
    root_path = Path(root_dir).resolve()
    original = geredata(fold, root_path)

    output_dir = root_path / "FasterRCNN" / fold
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_images = output_dir / "train"
    output_json = output_dir / "train_aug.json"

    _augment_coco_train(
        original.train_annotations,
        original.train_dir,
        output_images,
        output_json,
        copies=augmentation_copies,
    )

    return FasterDatasetConfig(
        root=root_path,
        train_dir=output_images,
        train_annotations=output_json,
        val_dir=original.val_dir,
        val_annotations=original.val_annotations,
    )


__all__ = ["GeraDobrasRCNN"]
