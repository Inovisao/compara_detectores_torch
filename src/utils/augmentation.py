from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2

_HSV_H: float = 0.015
_HSV_S: float = 0.7
_HSV_V: float = 0.4


def build_augmentation_pipeline(bbox_format: str) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=round(_HSV_H * 180),
                sat_shift_limit=round(_HSV_S * 100),
                val_shift_limit=round(_HSV_V * 100),
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format=bbox_format,
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


_YOLO_PIPELINE = build_augmentation_pipeline("yolo")


def _sanitize_yolo_bbox(cx: float, cy: float, w: float, h: float) -> list[float] | None:
    x_min = max(0.0, cx - w / 2)
    y_min = max(0.0, cy - h / 2)
    x_max = min(1.0, cx + w / 2)
    y_max = min(1.0, cy + h / 2)
    if x_max <= x_min or y_max <= y_min:
        return None
    return [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]


def read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return [], []
    class_ids, bboxes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            sanitized = _sanitize_yolo_bbox(*[float(v) for v in parts[1:]])
            if sanitized is not None:
                class_ids.append(int(parts[0]))
                bboxes.append(sanitized)
    return class_ids, bboxes


def write_yolo_labels(label_path: Path, class_ids: list[int], bboxes: list[list[float]]) -> None:
    lines = [
        f"{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n"
        for cid, b in zip(class_ids, bboxes)
    ]
    label_path.write_text("".join(lines), encoding="utf-8")


def augment_yolo_train_split(images_dir: Path, labels_dir: Path, copies: int) -> None:
    image_paths = sorted({
        p for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
        for p in images_dir.glob(ext)
    })

    for image_path in image_paths:
        class_ids, bboxes = read_yolo_labels(labels_dir / f"{image_path.stem}.txt")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for i in range(copies):
            result = _YOLO_PIPELINE(image=image_rgb, bboxes=bboxes, class_labels=class_ids)
            aug_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)

            suffix = f"_aug{i + 1}"
            cv2.imwrite(str(images_dir / f"{image_path.stem}{suffix}{image_path.suffix}"), aug_bgr)
            write_yolo_labels(
                labels_dir / f"{image_path.stem}{suffix}.txt",
                list(result["class_labels"]),
                [list(b) for b in result["bboxes"]],
            )
