from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


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
            image_dir = root / norm if norm in {"val", "test"} else root / "train"
            splits.append((norm, json_path, image_dir))
        return splits

    for candidate in ("train", "val", "valid", "test"):
        split_dir = root / candidate
        json_path = split_dir / "_annotations.coco.json"
        if json_path.exists():
            splits.append((_normalize_split(candidate), json_path, split_dir))

    if not splits:
        raise FileNotFoundError(
            "Unable to locate annotation files. "
            f"Expected a 'filesJSON' directory or split folders in {root}"
        )
    return splits


def _collect_class_names(annotation_paths: Sequence[Path]) -> List[str]:
    categories: Dict[int, str] = {}
    for path in annotation_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for category in data.get("categories", []):
            categories[int(category["id"])] = category["name"]

    if not categories:
        raise ValueError("No categories found in the provided annotation files.")
    return [categories[idx] for idx in sorted(categories)]


def _prepare_output_dirs(output_root: Path, splits: Iterable[str]) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in set(splits):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def _annotation_index(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    indexed: Dict[int, List[Dict]] = {}
    for ann in annotations:
        indexed.setdefault(int(ann["image_id"]), []).append(ann)
    return indexed


def _convert_bbox(ann: Dict, image_width: float, image_height: float) -> str:
    x, y, w, h = ann["bbox"]
    x_center = (x + w / 2.0) / image_width
    y_center = (y + h / 2.0) / image_height
    rel_width = w / image_width
    rel_height = h / image_height
    class_index = int(ann["category_id"]) - 1
    return f"{class_index} {x_center:.6f} {y_center:.6f} {rel_width:.6f} {rel_height:.6f}\n"


def _process_split(
    split_name: str,
    json_path: Path,
    images_src: Path,
    output_root: Path,
    dataset_root: Path,
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    annotations_by_image = _annotation_index(dataset.get("annotations", []))
    labels_dir = output_root / split_name / "labels"
    images_dir = output_root / split_name / "images"

    for image_info in dataset.get("images", []):
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]
        image_width = float(image_info.get("width", 1)) or 1.0
        image_height = float(image_info.get("height", 1)) or 1.0

        label_lines = [
            _convert_bbox(ann, image_width, image_height)
            for ann in annotations_by_image.get(image_id, [])
        ]

        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.writelines(label_lines)

        source_image = images_src / file_name
        if not source_image.exists():
            fallback = dataset_root / "train" / file_name
            if fallback.exists():
                source_image = fallback
            else:
                raise FileNotFoundError(
                    f"Image '{file_name}' referenced in {json_path} not found in "
                    f"{images_src} or fallback {fallback}"
                )

        shutil.copy(source_image, images_dir / file_name)


def _write_data_yaml(
    data_yaml_path: Path,
    output_root: Path,
    class_names: Sequence[str],
    splits: Iterable[str],
) -> None:
    splits_set = set(splits)
    required = {"train", "val"}
    if not required.issubset(splits_set):
        raise ValueError(f"Missing required splits {required - splits_set} to build data.yaml")

    content = {
        "train": str((output_root / "train" / "images").resolve()),
        "val": str((output_root / "val" / "images").resolve()),
        "nc": len(class_names),
        "names": list(class_names),
    }

    if "test" in splits_set:
        content["test"] = str((output_root / "test" / "images").resolve())

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=False)


def CriarLabelsYOLOV8(fold: str, root_dir: str | Path) -> Path:
    dataset_root = Path(root_dir).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits_info = _resolve_split_paths(dataset_root, fold)
    split_names = [info[0] for info in splits_info]
    annotation_paths = [info[1] for info in splits_info]
    class_names = _collect_class_names(annotation_paths)

    output_root = dataset_root / "YOLO"
    _prepare_output_dirs(output_root, split_names)

    for split_name, json_path, images_src in splits_info:
        _process_split(split_name, json_path, images_src, output_root, dataset_root)

    data_yaml_path = dataset_root / "data_yolov8.yaml"
    _write_data_yaml(data_yaml_path, output_root, class_names, split_names)
    return data_yaml_path


__all__ = ["CriarLabelsYOLOV8"]
