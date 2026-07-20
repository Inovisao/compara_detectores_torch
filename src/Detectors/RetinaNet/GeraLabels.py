from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataset_contract import split_image_dir


@dataclass(frozen=True)
class SplitPaths:
    images: Path
    annotations: Path


@dataclass(frozen=True)
class RetinaNetDatasetConfig:
    train: SplitPaths
    val: SplitPaths
    test: Optional[SplitPaths]
    class_names: List[str]
    category_mapping: Dict[int, int]


def _normalize_split(name: str) -> str:
    normalized = name.lower()
    if normalized in {"val", "valid", "validation"}:
        return "val"
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"test", "testing"}:
        return "test"
    return normalized


def _resolve_split_paths(root: Path, fold: str) -> List[Tuple[str, Path, Path]]:
    files_json_dir = root / "filesJSON"
    splits: List[Tuple[str, Path, Path]] = []

    if files_json_dir.exists():
        json_paths = sorted(p for p in files_json_dir.glob(f"{fold}_*.json") if p.is_file())
        if not json_paths:
            raise FileNotFoundError(f"No JSON splits found for fold '{fold}' in {files_json_dir}")
        for json_path in json_paths:
            split_token = json_path.stem.split("_")[-1]
            norm = _normalize_split(split_token)
            image_dir = split_image_dir(root, norm, fold)
            splits.append((norm, json_path, image_dir))
        return splits

    raise FileNotFoundError(f"Expected filesJSON/ in DATASET_ROOT: {root}")


def _collect_categories(annotation_paths: List[Path]) -> Tuple[List[str], Dict[int, int]]:
    categories: Dict[int, str] = {}
    used_ids: set = set()
    for path in annotation_paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for category in data.get("categories", []):
            categories[int(category["id"])] = category["name"]
        for ann in data.get("annotations", []):
            used_ids.add(int(ann["category_id"]))

    # Ignora categorias definidas mas sem nenhuma anotação (classe fantasma)
    categories = {k: v for k, v in categories.items() if k in used_ids}

    if not categories:
        raise ValueError("No categories found in the provided annotation files.")

    sorted_items = sorted(categories.items())
    class_names = [name for _, name in sorted_items]
    mapping = {category_id: idx + 1 for idx, (category_id, _) in enumerate(sorted_items)}
    return class_names, mapping


def CriarLabelsRetinaNet(fold: str, root_dir: str | Path) -> RetinaNetDatasetConfig:
    dataset_root = Path(root_dir).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits_info = _resolve_split_paths(dataset_root, fold)
    split_index = {name: (annotations, images_src) for name, annotations, images_src in splits_info}

    if "train" not in split_index or "val" not in split_index:
        raise ValueError("RetinaNet training requires at least 'train' and 'val' splits.")

    class_names, mapping = _collect_categories([info[0] for info in split_index.values()])

    def _build_split(name: str) -> SplitPaths:
        annotations, images_src = split_index[name]
        return SplitPaths(images=images_src, annotations=annotations)

    train_split = _build_split("train")
    val_split = _build_split("val")
    test_split = _build_split("test") if "test" in split_index else None

    return RetinaNetDatasetConfig(
        train=train_split,
        val=val_split,
        test=test_split,
        class_names=class_names,
        category_mapping=mapping,
    )


__all__ = ["CriarLabelsRetinaNet", "RetinaNetDatasetConfig", "SplitPaths"]
