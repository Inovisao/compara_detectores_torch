"""COCO-format PyTorch Dataset."""

import json
from pathlib import Path
from typing import Optional

import cv2
import torch


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_json: str,
        images_dir: str,
        image_ids: Optional[list[int]] = None,
        transforms=None,
    ):
        with open(coco_json, "r") as f:
            coco = json.load(f)

        self.images_dir = Path(images_dir)
        self.transforms = transforms

        self.images = {img["id"]: img for img in coco["images"]}
        if image_ids is not None:
            self.images = {k: v for k, v in self.images.items() if k in image_ids}

        self._anns_by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.images:
                self._anns_by_image.setdefault(img_id, []).append(ann)

        self.image_ids = sorted(self.images.keys())
        self.num_classes = len(coco["categories"])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        image = cv2.imread(str(self.images_dir / img_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self._anns_by_image.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        return image, target
