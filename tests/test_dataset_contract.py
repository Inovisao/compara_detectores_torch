from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from dataset_contract import (
    load_contract,
    resolve_evaluation_tiling_mode,
    split_image_dir,
    validate_dataset_contract,
)


def _write_coco(path: Path, file_name: str = "tile.jpg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": file_name, "width": 640, "height": 320}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        }],
        "categories": [{"id": 1, "name": "insect"}],
    }), encoding="utf-8")


def _write_manifest(root: Path, evaluation_mode: str = "basic") -> None:
    root.joinpath("dataset_manifest.json").write_text(json.dumps({
        "contract_version": "1.0",
        "dataset_name": "asahi_rect",
        "dataset_type": "tiled_detection",
        "annotation_format": "coco",
        "splits": ["train", "val", "test"],
        "folds": ["fold_1"],
        "layout": {
            "annotations_dir": "filesJSON",
            "annotation_pattern": "{fold}_{split}.json",
            "image_dir_pattern": "{fold}/{split}/images",
            "label_dir_pattern": "{fold}/{split}/labels",
        },
        "tiling": {"mode": "asahi_rect", "evaluation_mode": evaluation_mode},
        "classes": [{"id": 1, "name": "insect"}],
    }), encoding="utf-8")


def test_manifest_controls_split_dirs_and_tiling_mode(tmp_path):
    _write_manifest(tmp_path, evaluation_mode="basic")

    contract = load_contract(tmp_path)

    assert contract is not None
    assert split_image_dir(tmp_path, "val", "fold_1") == tmp_path / "fold_1" / "val" / "images"
    assert resolve_evaluation_tiling_mode(tmp_path) == "basic"


def test_requested_tiling_mode_overrides_manifest(tmp_path):
    _write_manifest(tmp_path, evaluation_mode="basic")

    assert resolve_evaluation_tiling_mode(tmp_path, "sage") == "sage"


def test_validate_dataset_contract_accepts_complete_asahi_rect_layout(tmp_path):
    _write_manifest(tmp_path)
    for split in ("train", "val", "test"):
        split_dir = tmp_path / "fold_1" / split / "images"
        split_dir.mkdir(parents=True)
        (split_dir / "tile.jpg").write_bytes(b"fake")
        _write_coco(tmp_path / "filesJSON" / f"fold_1_{split}.json")

    assert validate_dataset_contract(tmp_path) == []


def test_contract_without_manifest_still_requires_crossfold_layout(tmp_path):
    for split in ("train", "val", "test"):
        split_dir = tmp_path / "fold_1" / split / "images"
        split_dir.mkdir(parents=True)
        (split_dir / "tile.jpg").write_bytes(b"fake")
        _write_coco(tmp_path / "filesJSON" / f"fold_1_{split}.json")

    assert split_image_dir(tmp_path, "test", "fold_1") == tmp_path / "fold_1" / "test" / "images"
    assert validate_dataset_contract(tmp_path) == []


def test_contract_does_not_fallback_to_flat_split_dirs(tmp_path):
    (tmp_path / "test").mkdir()
    _write_coco(tmp_path / "filesJSON" / "fold_1_test.json")

    errors = validate_dataset_contract(tmp_path)

    assert any("fold_1/test/images" in error for error in errors)
