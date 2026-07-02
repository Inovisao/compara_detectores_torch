from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Detectors.YOLOV8 import GeraLabels
from Detectors.YOLOV8.GeraLabels import map_category_id_to_class_index, normalize_bbox_to_yolo


def test_root_data_dir_is_repo_relative():
    expected = (Path(GeraLabels.__file__).resolve().parents[3] / "dataset" / "all").resolve()
    assert Path(GeraLabels.ROOT_DATA_DIR).resolve() == expected


def test_map_category_id_to_class_index_for_single_class():
    assert map_category_id_to_class_index(1, ["insetos"]) == 0


def test_normalize_bbox_to_yolo_uses_image_dimensions():
    bbox = [1362, 2652, 104.98, 94.58]

    normalized = normalize_bbox_to_yolo(bbox, 3072, 4080)

    assert normalized == pytest.approx([0.460, 0.661, 0.0342, 0.0232], rel=1e-3)
    assert max(normalized) <= 1.0
