from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_DATA_DIR = REPO_ROOT / 'dataset' / 'all'
FILES_JSON_DIR = ROOT_DATA_DIR / 'filesJSON'
YOLO_OUTPUT_DIR = ROOT_DATA_DIR / 'YOLOV5_TPH'
DATA_YAML_PATH = ROOT_DATA_DIR / 'data_yolov5_tph.yaml'

def _get_use_tiled_dataset():
    """Check if tiled dataset mode is enabled"""
    import os
    return os.getenv('USE_TILED_DATASET', 'true').lower() == 'true'

def _load_categories(root_data_dir: Path = None) -> List[str]:
    if root_data_dir is None:
        root_data_dir = ROOT_DATA_DIR
    annotations_path = root_data_dir / 'train' / '_annotations.coco.json'
    if not annotations_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_path}")

    with open(annotations_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    category_ids = {ann['category_id'] for ann in dataset.get('annotations', [])}
    return [
        category['name']
        for category in dataset.get('categories', [])
        if category['id'] in category_ids
    ]


def _write_data_yaml(class_names: List[str], yolo_output_dir: Path, data_yaml_path: Path) -> None:
    content = {
        'train': str((yolo_output_dir / 'train' / 'images').resolve()),
        'val': str((yolo_output_dir / 'val' / 'images').resolve()),
        'test': str((yolo_output_dir / 'test' / 'images').resolve()),
        'nc': len(class_names),
        'names': class_names,
    }

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def _prepare_output_dirs(yolo_output_dir: Path) -> None:
    if yolo_output_dir.exists():
        shutil.rmtree(yolo_output_dir)
    for split in ('train', 'val', 'test'):
        (yolo_output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)


def _iter_fold_jsons(fold: str, root_data_dir: Path) -> Iterable[Path]:
    files_json_dir = root_data_dir / 'filesJSON'
    if not files_json_dir.exists():
        raise FileNotFoundError(f"filesJSON directory not found: {files_json_dir}")

    prefix = f"{fold}_"
    paths = sorted(p for p in files_json_dir.glob(f"{fold}_*.json") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No JSON splits found for fold '{fold}' in {files_json_dir}")
    return paths


def _annotation_index(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    indexed: Dict[int, List[Dict]] = {}
    for ann in annotations:
        image_id = int(ann['image_id'])
        indexed.setdefault(image_id, []).append(ann)
    return indexed


def _convert_bbox(ann: Dict, image_width: float, image_height: float) -> str:
    x, y, w, h = ann['bbox']
    x_center = (x + w / 2.0) / image_width
    y_center = (y + h / 2.0) / image_height
    rel_width = w / image_width
    rel_height = h / image_height
    class_index = int(ann['category_id']) - 1
    return f"{class_index} {x_center:.6f} {y_center:.6f} {rel_width:.6f} {rel_height:.6f}\n"


def _process_split(split_name: str, json_path: Path, root_data_dir: Path, yolo_output_dir: Path, use_tiled: bool) -> None:
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    annotations_by_image = _annotation_index(dataset.get('annotations', []))
    labels_dir = yolo_output_dir / split_name / 'labels'
    images_dir = yolo_output_dir / split_name / 'images'

    for image_info in dataset.get('images', []):
        image_id = int(image_info['id'])
        file_name = image_info['file_name']
        image_width = float(image_info.get('width', 1))
        image_height = float(image_info.get('height', 1))

        label_lines = [
            _convert_bbox(ann, image_width, image_height)
            for ann in annotations_by_image.get(image_id, [])
        ]

        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        with open(label_path, 'w', encoding='utf-8') as label_file:
            label_file.writelines(label_lines)

        # Determine source image location
        if use_tiled:
            # Images are in the same directory as annotations
            source_image = root_data_dir / split_name / file_name
        else:
            # Images are in the train folder
            source_image = root_data_dir / 'train' / file_name

        if not source_image.exists():
            raise FileNotFoundError(f"Image referenced in annotations not found: {source_image}")
        shutil.copy(source_image, images_dir / file_name)


def CriarLabelsYOLOV5TPH(fold: str, root_data_dir_str: str = None) -> None:
    # Convert to Path
    if root_data_dir_str:
        root_data_dir = Path(root_data_dir_str)
    else:
        root_data_dir = ROOT_DATA_DIR

    use_tiled = _get_use_tiled_dataset()

    # Set output directories
    yolo_output_dir = root_data_dir / 'YOLOV5_TPH'
    data_yaml_path = root_data_dir / 'data_yolov5_tph.yaml'

    class_names = _load_categories(root_data_dir)
    _prepare_output_dirs(yolo_output_dir)
    _write_data_yaml(class_names, yolo_output_dir, data_yaml_path)

    if use_tiled:
        # For tiled datasets, read directly from train/val/test folders
        for split in ['train', 'val', 'test']:
            split_dir = root_data_dir / split
            annotations_path = split_dir / '_annotations.coco.json'
            if annotations_path.exists():
                _process_split(split, annotations_path, root_data_dir, yolo_output_dir, use_tiled)
    else:
        # Original logic: read from filesJSON
        for json_path in _iter_fold_jsons(fold, root_data_dir):
            split = json_path.stem.split('_')[-1]
            if split == 'valid':
                split = 'val'
            _process_split(split, json_path, root_data_dir, yolo_output_dir, use_tiled)
