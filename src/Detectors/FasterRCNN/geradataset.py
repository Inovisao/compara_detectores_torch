from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
from dataset_contract import split_image_dir


@dataclass(frozen=True)
class FasterDatasetConfig:
    root: Path
    train_dir: Path
    train_annotations: Path
    val_dir: Path
    val_annotations: Path


def _resolve_from_filesjson(fold: str, dataset_root: Path) -> Optional[FasterDatasetConfig]:
    files_json_dir = dataset_root / "filesJSON"
    if not files_json_dir.exists():
        return None

    train_ann = files_json_dir / f"{fold}_train.json"
    val_ann = files_json_dir / f"{fold}_val.json"
    train_dir = split_image_dir(dataset_root, "train", fold)
    val_dir = split_image_dir(dataset_root, "val", fold)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train images directory not found for FasterRCNN: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation images directory not found for FasterRCNN: {val_dir}")
    if not train_ann.exists():
        raise FileNotFoundError(f"Train annotations not found for fold '{fold}': {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"Validation annotations not found for fold '{fold}': {val_ann}")

    return FasterDatasetConfig(
        root=dataset_root,
        train_dir=train_dir,
        train_annotations=train_ann,
        val_dir=val_dir,
        val_annotations=val_ann,
    )


def _find_split_dir(dataset_root: Path, candidates: Tuple[str, ...]) -> Tuple[Path, Path]:
    for name in candidates:
        split_dir = dataset_root / name
        annotations = split_dir / "_annotations.coco.json"
        if split_dir.exists() and annotations.exists():
            return split_dir, annotations
    raise FileNotFoundError(
        f"Unable to locate split directory with annotations under {dataset_root}. "
        f"Tried: {', '.join(str(dataset_root / c) for c in candidates)}"
    )


def _resolve_from_split_dirs(dataset_root: Path) -> FasterDatasetConfig:
    train_dir, train_ann = _find_split_dir(dataset_root, ("train", "training"))
    val_dir, val_ann = _find_split_dir(dataset_root, ("val", "valid", "validation"))

    return FasterDatasetConfig(
        root=dataset_root,
        train_dir=train_dir,
        train_annotations=train_ann,
        val_dir=val_dir,
        val_annotations=val_ann,
    )


def geredata(fold: str, dataset_root: Union[str, Path]) -> FasterDatasetConfig:
    root_path = Path(dataset_root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found for FasterRCNN: {root_path}")

    config = _resolve_from_filesjson(fold, root_path)
    if config:
        return config

    return _resolve_from_split_dirs(root_path)


__all__ = ["FasterDatasetConfig", "geredata"]
