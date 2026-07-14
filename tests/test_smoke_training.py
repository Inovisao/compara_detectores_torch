"""
Smoke tests para garantir que o pipeline de treino não quebra ao rodar:
  DATASET_ROOT=dataset/<ds> python src/main.py

Testa imports, resolução de paths e carregamento do config — sem treinar nada.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_contract import split_image_dir

DATASETS     = ["sahi", "asahi", "asahi_rect"]
FOLDS        = [f"fold_{i}" for i in range(1, 6)]


@pytest.fixture(autouse=True)
def src_on_path():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    yield
    # remove tph-yolov5 from sys.path to avoid polluting other tests
    sys.path[:] = [p for p in sys.path if "tph-yolov5" not in p]


@pytest.fixture(params=DATASETS)
def ds_root(request, tmp_path):
    root = PROJECT_ROOT / "dataset" / request.param
    if not root.exists():
        pytest.skip(f"Dataset not found: {root}")
    if not (root / "filesJSON").exists():
        pytest.skip(f"filesJSON not set up: {root}")
    return root


# ── DETR config ───────────────────────────────────────────────────────────────

class TestDetrConfig:
    def test_config_loads_with_dataset_root(self, ds_root):
        """DETR config.py deve ler _annotations.coco.json via DATASET_ROOT."""
        os.environ["DATASET_ROOT"] = str(ds_root)
        # force reimport since ROOT_DATA_DIR is module-level
        for key in list(sys.modules):
            if "Detectors.Detr.config" in key or key == "Detectors.Detr.config":
                del sys.modules[key]
        try:
            from Detectors.Detr import config as detr_cfg
            assert detr_cfg.NUM_CLASSES >= 2, \
                f"Expected at least 2 classes (background + insect), got {detr_cfg.NUM_CLASSES}"
            assert "__background__" in detr_cfg.CLASSES
        finally:
            del os.environ["DATASET_ROOT"]

    def test_annotations_json_exists(self, ds_root):
        ann = ds_root / "filesJSON" / "fold_1_train.json"
        assert ann.exists(), f"Missing {ann}"

    def test_annotations_json_parseable(self, ds_root):
        ann = ds_root / "filesJSON" / "fold_1_train.json"
        with open(ann) as f:
            data = json.load(f)
        assert "categories" in data and len(data["categories"]) > 0


# ── YOLOV8 label generation ───────────────────────────────────────────────────

class TestYOLOV8Labels:
    @pytest.mark.parametrize("fold", FOLDS)
    def test_resolve_split_paths(self, ds_root, fold):
        from Detectors.YOLOV8.GeraLabels import _resolve_split_paths
        splits = _resolve_split_paths(ds_root, fold)
        split_names = {s[0] for s in splits}
        missing = {"train", "val", "test"} - split_names
        assert not missing, f"{fold}: missing splits {missing}"

    @pytest.mark.parametrize("fold", FOLDS)
    def test_image_dirs_exist(self, ds_root, fold):
        from Detectors.YOLOV8.GeraLabels import _resolve_split_paths
        for split_name, json_path, image_dir in _resolve_split_paths(ds_root, fold):
            assert image_dir.exists(), \
                f"{fold}/{split_name}: image dir not found: {image_dir}"
            imgs = [p for p in image_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            assert len(imgs) > 0, f"{fold}/{split_name}: image dir is empty: {image_dir}"


# ── FasterRCNN dataset resolution ────────────────────────────────────────────

class TestFasterDataset:
    @pytest.mark.parametrize("fold", FOLDS)
    def test_geredata_resolves(self, ds_root, fold):
        from Detectors.FasterRCNN.geradataset import geredata
        cfg = geredata(fold, ds_root)
        assert cfg.train_dir.exists(),  f"{fold}: train_dir not found: {cfg.train_dir}"
        assert cfg.train_annotations.exists(), \
            f"{fold}: train_ann not found: {cfg.train_annotations}"
        assert cfg.val_dir.exists(),    f"{fold}: val_dir not found: {cfg.val_dir}"
        assert cfg.val_annotations.exists(), \
            f"{fold}: val_ann not found: {cfg.val_annotations}"


# ── DETR GeraDobras image lookup ──────────────────────────────────────────────

class TestDetrGeraDobras:
    @pytest.mark.parametrize("fold", FOLDS)
    def test_filesJSON_found_for_fold(self, ds_root, fold):
        fj_dir = ds_root / "filesJSON"
        for split in ("train", "val", "test"):
            p = fj_dir / f"{fold}_{split}.json"
            assert p.exists(), f"Missing: {p}"

    @pytest.mark.parametrize("fold", ["fold_1"])
    def test_val_images_findable(self, ds_root, fold):
        """Verifica que as imagens de val referenciadas no JSON existem em val/."""
        fj_path = ds_root / "filesJSON" / f"{fold}_val.json"
        with open(fj_path) as f:
            data = json.load(f)
        val_dir = split_image_dir(ds_root, "val", fold)
        missing = []
        for img in data["images"][:20]:  # amostra de 20
            fname = img["file_name"]
            if not (val_dir / fname).exists() and not (val_dir / fname.lower()).exists():
                missing.append(fname)
        assert not missing, f"{fold}/val: images not found in {val_dir}: {missing[:5]}"

    @pytest.mark.parametrize("fold", ["fold_1"])
    def test_train_tiles_findable(self, ds_root, fold):
        """Verifica que os tiles de train referenciados no JSON existem em train/."""
        fj_path = ds_root / "filesJSON" / f"{fold}_train.json"
        with open(fj_path) as f:
            data = json.load(f)
        train_dir = split_image_dir(ds_root, "train", fold)
        missing = []
        for img in data["images"][:30]:  # amostra de 30
            if not (train_dir / img["file_name"]).exists():
                missing.append(img["file_name"])
        assert not missing, f"{fold}/train: tiles not found in {train_dir}: {missing[:5]}"
