from __future__ import annotations

from pathlib import Path

from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from utils.augmentation import augment_yolo_train_split


def GeraDobrasYOLOV8(fold: str, root_dir: str | Path, augmentation_copies: int = 2) -> Path:
    data_yaml_path = CriarLabelsYOLOV8(fold, root_dir)

    dataset_root = Path(root_dir).resolve()
    augment_yolo_train_split(
        dataset_root / "YOLO" / "train" / "images",
        dataset_root / "YOLO" / "train" / "labels",
        copies=augmentation_copies,
    )

    return data_yaml_path


__all__ = ["GeraDobrasYOLOV8"]
