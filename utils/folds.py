"""Runtime cross-validation fold splitting from COCO images/annotations."""

import random
from sklearn.model_selection import train_test_split


def split_folds(
    images: list[dict],
    annotations: list[dict],
    n_folds: int,
    val_ratio: float,
    seed: int = 42,
) -> list[dict]:
    """Split COCO images into n_folds of (train, val, test) image_id sets.

    Returns list of dicts with 'train', 'val', 'test' keys, each a list of image_ids.
    Uses sequential splitting after shuffle for deterministic, non-overlapping test sets.
    """
    random.seed(seed)
    image_ids = [img["id"] for img in images]
    random.shuffle(image_ids)

    folds = []
    fold_size = len(image_ids) // n_folds
    remainder = len(image_ids) % n_folds

    start = 0
    for i in range(n_folds):
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        test_ids = image_ids[start:end]

        remaining_ids = [x for x in image_ids if x not in test_ids]
        train_ids, val_ids = train_test_split(
            remaining_ids, test_size=val_ratio, random_state=seed
        )
        folds.append({"train": train_ids, "val": val_ids, "test": test_ids})
        start = end

    return folds
