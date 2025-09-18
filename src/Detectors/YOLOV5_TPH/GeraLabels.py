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


def _load_categories() -> List[str]:
    annotations_path = ROOT_DATA_DIR / 'train' / '_annotations.coco.json'
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


def _write_data_yaml(class_names: List[str]) -> None:
    content = {
        'train': str((YOLO_OUTPUT_DIR / 'train' / 'images').resolve()),
        'val': str((YOLO_OUTPUT_DIR / 'val' / 'images').resolve()),
        'test': str((YOLO_OUTPUT_DIR / 'test' / 'images').resolve()),
        'nc': len(class_names),
        'names': class_names,
    }

    DATA_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def _prepare_output_dirs() -> None:
    if YOLO_OUTPUT_DIR.exists():
        shutil.rmtree(YOLO_OUTPUT_DIR)
    for split in ('train', 'val', 'test'):
        (YOLO_OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)


def _iter_fold_jsons(fold: str) -> Iterable[Path]:
    if not FILES_JSON_DIR.exists():
        raise FileNotFoundError(f"filesJSON directory not found: {FILES_JSON_DIR}")

    prefix = f"{fold}_"
    paths = sorted(p for p in FILES_JSON_DIR.glob(f"{fold}_*.json") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No JSON splits found for fold '{fold}' in {FILES_JSON_DIR}")
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


def _process_split(split_name: str, json_path: Path) -> None:
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    annotations_by_image = _annotation_index(dataset.get('annotations', []))
    labels_dir = YOLO_OUTPUT_DIR / split_name / 'labels'
    images_dir = YOLO_OUTPUT_DIR / split_name / 'images'

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

        source_image = ROOT_DATA_DIR / 'train' / file_name
        if not source_image.exists():
            raise FileNotFoundError(f"Image referenced in annotations not found: {source_image}")
        shutil.copy(source_image, images_dir / file_name)


def CriarLabelsYOLOV5TPH(fold: str) -> None:
    class_names = _load_categories()
    _prepare_output_dirs()
    _write_data_yaml(class_names)

    for json_path in _iter_fold_jsons(fold):
        split = json_path.stem.split('_')[-1]
        if split == 'valid':
            split = 'val'
        _process_split(split, json_path)
