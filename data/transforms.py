"""Albumentations-based transform pipelines returning torch tensors."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )
