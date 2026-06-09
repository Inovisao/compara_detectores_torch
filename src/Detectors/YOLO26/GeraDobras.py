from __future__ import annotations

from pathlib import Path

from Detectors.YOLO26.GeraLabels import CriarLabelsYOLO26
from utils.augmentation import augment_yolo_train_split


def GeraDobrasYOLO26(fold: str, root_dir: str | Path, augmentation_copies: int = 2) -> Path:
    data_yaml_path = CriarLabelsYOLO26(fold, root_dir)

    dataset_root = Path(root_dir).resolve()
    augment_yolo_train_split(
        dataset_root / "YOLO26" / "train" / "images",
        dataset_root / "YOLO26" / "train" / "labels",
        copies=augmentation_copies,
    )

    return data_yaml_path


__all__ = ["GeraDobrasYOLO26"]
