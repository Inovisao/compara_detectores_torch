"""Test fold splitting logic."""
from utils.folds import split_folds


def test_split_folds_basic():
    images = [{"id": i} for i in range(100)]
    annotations = []
    folds = split_folds(images, annotations, n_folds=5, val_ratio=0.2, seed=42)

    assert len(folds) == 5
    for f in folds:
        assert "train" in f and "val" in f and "test" in f
        assert len(f["train"]) > 0
        assert len(f["val"]) > 0
        assert len(f["test"]) > 0

    all_test_ids = set()
    for f in folds:
        all_test_ids.update(f["test"])
    assert all_test_ids == set(range(100))


def test_split_folds_reproducible():
    images = [{"id": i} for i in range(20)]
    folds1 = split_folds(images, [], n_folds=3, val_ratio=0.3, seed=42)
    folds2 = split_folds(images, [], n_folds=3, val_ratio=0.3, seed=42)
    for f1, f2 in zip(folds1, folds2):
        assert f1["test"] == f2["test"]
