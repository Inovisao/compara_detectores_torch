from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.transforms.functional import to_tensor

from Detectors.RetinaNet.GeraLabels import (
    CriarLabelsRetinaNet,
    RetinaNetDatasetConfig,
    SplitPaths,
)
from Detectors.RetinaNet.config import get_config


class _CocoDataset(Dataset):
    def __init__(
        self,
        split: SplitPaths,
        category_mapping: Dict[int, int],
    ) -> None:
        self.images_dir = Path(split.images)
        self.annotations_path = Path(split.annotations)
        self.category_mapping = category_mapping

        with open(self.annotations_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.images_info = sorted(data.get("images", []), key=lambda img: img["id"])
        self.annotations_by_image: Dict[int, List[Dict]] = {}
        for ann in data.get("annotations", []):
            self.annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    def __len__(self) -> int:
        return len(self.images_info)

    def __getitem__(self, idx: int):
        sample = self.images_info[idx]
        image_id = int(sample["id"])
        file_name = sample["file_name"]
        width = float(sample.get("width", 1)) or 1.0
        height = float(sample.get("height", 1)) or 1.0

        image_path = self.images_dir / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found for RetinaNet training: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor(image)

        annotations = self.annotations_by_image.get(image_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x2 = x + w
            y2 = y + h
            mapped_label = self.category_mapping.get(int(ann["category_id"]))
            if mapped_label is None or w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x2, y2])
            labels.append(mapped_label)
            areas.append(float(ann.get("area", w * h)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas if areas else [0.0], dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if not boxes:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        return image_tensor, target


def _collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _prepare_dataloaders(config: RetinaNetDatasetConfig):
    train_dataset = _CocoDataset(config.train, config.category_mapping)
    val_dataset = _CocoDataset(config.val, config.category_mapping)

    cfg = get_config()
    print(
        f"[RetinaNet] Preparando dataloaders - train={len(train_dataset)} imgs ({config.train.images}), "
        f"val={len(val_dataset)} imgs ({config.val.images}), classes={len(config.class_names)}",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader, cfg


def _to_device(items, device):
    if isinstance(items, torch.Tensor):
        return items.to(device)
    if isinstance(items, list):
        return [_to_device(item, device) for item in items]
    if isinstance(items, dict):
        return {key: _to_device(value, device) for key, value in items.items()}
    return items


def runRetinaNet(fold: str, fold_dir: str, root_data_dir: str | Path) -> None:
    dataset_config = CriarLabelsRetinaNet(fold, root_data_dir)
    train_loader, val_loader, hyperparams = _prepare_dataloaders(dataset_config)

    num_classes = len(dataset_config.class_names) + 1  # include background class
    _ = RetinaNet_ResNet50_FPN_Weights.DEFAULT  # ensure weights are downloaded for transforms if needed
    backbone_weights = ResNet50_Weights.IMAGENET1K_V2
    model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes, weights_backbone=backbone_weights)

    device_str = hyperparams.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)
    print(
        f"[RetinaNet] Iniciando treino | device={device} | epochs={hyperparams.epochs} "
        f"| batch={hyperparams.batch_size} | lr={hyperparams.learning_rate}",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=hyperparams.learning_rate,
        momentum=hyperparams.momentum,
        weight_decay=hyperparams.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hyperparams.lr_step_size,
        gamma=hyperparams.lr_gamma,
    )

    best_state = None
    best_loss = float("inf")

    for epoch in range(hyperparams.epochs):
        print(f"[RetinaNet] Época {epoch + 1}/{hyperparams.epochs} - treinamento", flush=True)
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = _to_device(images, device)
            targets = _to_device(targets, device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(
                    f"[RetinaNet]   batch {batch_idx}/{len(train_loader)} "
                    f"loss={losses.item():.4f} "
                    f"(cls={loss_dict['classification'].item():.4f}, "
                    f"bbox={loss_dict['bbox_regression'].item():.4f})",
                    flush=True,
                )

        model.eval()
        print(f"[RetinaNet] Época {epoch + 1}/{hyperparams.epochs} - validação", flush=True)
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader, start=1):
                images = _to_device(images, device)
                targets = _to_device(targets, device)
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
                val_loss += losses.item()
                val_batches += 1
                if batch_idx == 1 or batch_idx % 10 == 0:
                    print(
                        f"[RetinaNet]   val batch {batch_idx}/{len(val_loader)} loss={losses.item():.4f}",
                        flush=True,
                    )

        avg_val_loss = val_loss / max(1, val_batches)
        print(
            f"[RetinaNet] Época {epoch + 1}: train_loss={running_loss / max(1, len(train_loader)):.4f} "
            f"| val_loss={avg_val_loss:.4f}",
            flush=True,
        )
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "class_names": dataset_config.class_names,
                "category_mapping": dataset_config.category_mapping,
            }

        lr_scheduler.step()

    if best_state is None:
        best_state = {
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "class_names": dataset_config.class_names,
        }

    target_dir = Path(fold_dir) / "RetinaNet"
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, target_dir / "best.pth")
    print(f"[RetinaNet] Treinamento concluído | melhor val_loss={best_loss:.4f}", flush=True)
