from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "all":          PROJECT_ROOT / "dataset" / "all",
    "fine_tuning":  PROJECT_ROOT / "dataset" / "fine_tuning",
}
EXPECTED_SPLITS  = {"train", "val", "test"}
EXPECTED_FOLDS   = {f"fold_{i}" for i in range(1, 6)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(params=list(DATASETS.keys()))
def ds(request):
    root = DATASETS[request.param]
    if not root.exists():
        pytest.skip(f"Dataset not found: {root}")
    return root


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _all_jsons(ds_root: Path) -> list[Path]:
    return sorted((ds_root / "filesJSON").glob("fold_*_*.json"))


def _images_on_disk(ds_root: Path) -> set[str]:
    train_dir = ds_root / "train"
    return {p.name for p in train_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS}


# ── structure ─────────────────────────────────────────────────────────────────

class TestStructure:
    def test_root_exists(self, ds):
        assert ds.exists()

    def test_filesjson_dir_exists(self, ds):
        assert (ds / "filesJSON").is_dir()

    def test_train_dir_exists(self, ds):
        assert (ds / "train").exists()

    def test_train_dir_not_empty(self, ds):
        imgs = [p for p in (ds / "train").iterdir() if p.suffix in IMAGE_EXTENSIONS]
        assert len(imgs) > 0, "train/ has no images"

    def test_all_folds_present(self, ds):
        found = {
            "_".join(p.stem.split("_")[:2])
            for p in _all_jsons(ds)
        }
        missing = EXPECTED_FOLDS - found
        assert not missing, f"Folds missing: {missing}"

    def test_all_splits_per_fold(self, ds):
        for fold in EXPECTED_FOLDS:
            splits = {
                p.stem.split("_")[-1]
                for p in (ds / "filesJSON").glob(f"{fold}_*.json")
            }
            missing = EXPECTED_SPLITS - splits
            assert not missing, f"{fold} missing splits: {missing}"

    def test_json_count(self, ds):
        n = len(_all_jsons(ds))
        expected = len(EXPECTED_FOLDS) * len(EXPECTED_SPLITS)
        assert n == expected, f"Expected {expected} JSONs, found {n}"


# ── coco json format ──────────────────────────────────────────────────────────

class TestCocoFormat:
    def test_required_keys_present(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            for key in ("images", "annotations", "categories"):
                assert key in data, f"{path.name}: missing key '{key}'"

    def test_categories_not_empty(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            assert len(data["categories"]) > 0, f"{path.name}: no categories"

    def test_category_ids_are_positive(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            for cat in data["categories"]:
                assert cat["id"] > 0, f"{path.name}: category id {cat['id']} <= 0"

    def test_images_have_required_fields(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            for img in data["images"]:
                for field in ("id", "file_name", "width", "height"):
                    assert field in img, f"{path.name}: image missing field '{field}'"

    def test_image_dimensions_positive(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            for img in data["images"]:
                assert img["width"] > 0 and img["height"] > 0, \
                    f"{path.name}: image {img['file_name']} has non-positive dimensions"

    def test_annotations_have_required_fields(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            for ann in data["annotations"]:
                for field in ("id", "image_id", "category_id", "bbox"):
                    assert field in ann, f"{path.name}: annotation missing field '{field}'"

    def test_splits_not_empty(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            assert len(data["images"]) > 0, f"{path.name}: no images"

    def test_annotations_reference_valid_categories(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            valid_ids = {c["id"] for c in data["categories"]}
            for ann in data["annotations"]:
                assert ann["category_id"] in valid_ids, \
                    f"{path.name}: annotation {ann['id']} has unknown category_id {ann['category_id']}"

    def test_annotations_reference_valid_images(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            valid_ids = {img["id"] for img in data["images"]}
            for ann in data["annotations"]:
                assert ann["image_id"] in valid_ids, \
                    f"{path.name}: annotation {ann['id']} references unknown image_id {ann['image_id']}"


# ── bounding boxes ────────────────────────────────────────────────────────────

class TestBoundingBoxes:
    def test_bbox_has_four_values(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            for ann in data["annotations"]:
                assert len(ann["bbox"]) == 4, \
                    f"{path.name}: annotation {ann['id']} bbox has {len(ann['bbox'])} values"

    def test_bbox_positive_dimensions(self, ds):
        bad = []
        for path in _all_jsons(ds):
            data = _load_json(path)
            for ann in data["annotations"]:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    bad.append((path.name, ann["id"], w, h))
        assert not bad, f"Bboxes with non-positive w/h: {bad[:5]}"

    def test_bbox_origin_non_negative(self, ds):
        bad = []
        for path in _all_jsons(ds):
            data = _load_json(path)
            for ann in data["annotations"]:
                x, y, w, h = ann["bbox"]
                if x < 0 or y < 0:
                    bad.append((path.name, ann["id"], x, y))
        assert not bad, f"Bboxes with negative origin: {bad[:5]}"

    def test_bbox_within_image_bounds(self, ds):
        bad = []
        for path in _all_jsons(ds):
            data = _load_json(path)
            img_dims = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
            for ann in data["annotations"]:
                x, y, w, h = ann["bbox"]
                iw, ih = img_dims[ann["image_id"]]
                tol = 1.0
                if x + w > iw + tol or y + h > ih + tol:
                    bad.append((path.name, ann["id"], x, y, w, h, iw, ih))
        assert not bad, f"Bboxes outside image bounds: {bad[:5]}"


# ── image files on disk ───────────────────────────────────────────────────────

class TestImageFiles:
    def test_all_referenced_images_exist(self, ds):
        on_disk = _images_on_disk(ds)
        missing: list[tuple[str, str]] = []
        for path in _all_jsons(ds):
            data = _load_json(path)
            for img in data["images"]:
                if img["file_name"] not in on_disk:
                    missing.append((path.name, img["file_name"]))
        assert not missing, f"Missing images ({len(missing)} total): {missing[:5]}"

    def test_no_duplicate_image_ids_within_split(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            ids = [img["id"] for img in data["images"]]
            dupes = [i for i in set(ids) if ids.count(i) > 1]
            assert not dupes, f"{path.name}: duplicate image ids: {dupes}"

    def test_no_duplicate_annotation_ids_within_split(self, ds):
        for path in _all_jsons(ds):
            data = _load_json(path)
            ids = [ann["id"] for ann in data["annotations"]]
            dupes = [i for i in set(ids) if ids.count(i) > 1]
            assert not dupes, f"{path.name}: duplicate annotation ids: {dupes}"


# ── category consistency across folds ─────────────────────────────────────────

class TestCategoryConsistency:
    def test_same_categories_across_all_folds(self, ds):
        reference = None
        ref_name = None
        for path in _all_jsons(ds):
            data = _load_json(path)
            cats = {(c["id"], c["name"]) for c in data["categories"]}
            if reference is None:
                reference = cats
                ref_name = path.name
            else:
                assert cats == reference, \
                    f"{path.name} categories differ from {ref_name}: " \
                    f"added={cats - reference}, removed={reference - cats}"


# ── train / test leakage ──────────────────────────────────────────────────────

class TestNoLeakage:
    def test_no_image_in_both_train_and_test(self, ds):
        for fold in EXPECTED_FOLDS:
            train_path = ds / "filesJSON" / f"{fold}_train.json"
            test_path  = ds / "filesJSON" / f"{fold}_test.json"
            if not train_path.exists() or not test_path.exists():
                continue
            train_files = {img["file_name"] for img in _load_json(train_path)["images"]}
            test_files  = {img["file_name"] for img in _load_json(test_path)["images"]}
            overlap = train_files & test_files
            assert not overlap, \
                f"{fold}: {len(overlap)} images in both train and test: {list(overlap)[:3]}"

    def test_no_image_in_both_train_and_val(self, ds):
        for fold in EXPECTED_FOLDS:
            train_path = ds / "filesJSON" / f"{fold}_train.json"
            val_path   = ds / "filesJSON" / f"{fold}_val.json"
            if not train_path.exists() or not val_path.exists():
                continue
            train_files = {img["file_name"] for img in _load_json(train_path)["images"]}
            val_files   = {img["file_name"] for img in _load_json(val_path)["images"]}
            overlap = train_files & val_files
            assert not overlap, \
                f"{fold}: {len(overlap)} images in both train and val: {list(overlap)[:3]}"
