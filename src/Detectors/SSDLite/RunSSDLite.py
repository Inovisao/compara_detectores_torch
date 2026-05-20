from __future__ import annotations

import json
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.transforms.functional import to_tensor

from Detectors.SSDLite.GeraLabels import (
    CriarLabelsSSDLite,
    SSDLiteDatasetConfig,
    SplitPaths,
)
from Detectors.SSDLite.config import get_config

# ── Feature map indices in MobileNetV2 for a 320×320 input ──────────────────
# features[13] → 20×20×96   (diagram: Camada 14)
# features[17] → 10×10×320  (diagram: Camada 19)
_STAGE1_END = 14   # slice [0:14] → 20×20×96
_STAGE2_END = 18   # slice [14:18] → 10×10×320

# SSDLite auxiliary layer output channels per extra scale
_AUX_CHANNELS = [256, 256, 256, 64]

# All 6 feature map channel counts fed into the SSD head
_BACKBONE_OUT_CHANNELS = [96, 320] + _AUX_CHANNELS


# GroupNorm with 32 groups works for all channel counts used here (64, 96, 256, 320)
# and unlike BatchNorm2d it is stable at any spatial size including 1×1,
# which is reached by the last auxiliary layer.
_NORM_LAYER: Callable[..., nn.Module] = partial(nn.GroupNorm, 32)


def _dw_sep_conv(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    """Depthwise-separable conv block as used in SSDLite auxiliary layers."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        _NORM_LAYER(in_ch),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        _NORM_LAYER(out_ch),
        nn.ReLU6(inplace=True),
    )


class _MobileNetV2SSDLiteBackbone(nn.Module):
    """
    MobileNetV2 backbone with SSDLite auxiliary feature pyramid.

    Produces 6 feature maps (20×20, 10×10, 5×5, 3×3, 2×2, 1×1) from a
    320×320 input, matching the architecture in the diagram.
    """

    out_channels: List[int] = _BACKBONE_OUT_CHANNELS

    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        self.stage1 = backbone.features[:_STAGE1_END]   # → 20×20×96
        self.stage2 = backbone.features[_STAGE1_END:_STAGE2_END]  # → 10×10×320

        prev_ch = 320
        aux_layers = []
        for out_ch in _AUX_CHANNELS:
            aux_layers.append(_dw_sep_conv(prev_ch, out_ch, stride=2))
            prev_ch = out_ch
        self.aux_layers = nn.ModuleList(aux_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.stage1(x)   # 20×20×96
        f2 = self.stage2(f1)  # 10×10×320

        features = [f1, f2]
        feat = f2
        for layer in self.aux_layers:
            feat = layer(feat)
            features.append(feat)

        # SSD expects an OrderedDict with string keys
        return OrderedDict((str(i), f) for i, f in enumerate(features))


def _build_model(num_classes: int, cfg) -> SSD:
    backbone = _MobileNetV2SSDLiteBackbone()

    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    head = SSDLiteHead(
        in_channels=_BACKBONE_OUT_CHANNELS,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=_NORM_LAYER,
    )

    return SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(320, 320),
        num_classes=num_classes,
        head=head,
        score_thresh=cfg.score_thresh,
        nms_thresh=cfg.nms_thresh,
        detections_per_img=cfg.detections_per_img,
    )


class _CocoDataset(Dataset):
    def __init__(self, split: SplitPaths, category_mapping: Dict[int, int]) -> None:
        self.images_dir = Path(split.images)
        self.category_mapping = category_mapping

        with open(split.annotations, "r", encoding="utf-8") as handle:
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

        image_path = self.images_dir / sample["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor(image)

        annotations = self.annotations_by_image.get(image_id, [])
        boxes, labels, areas = [], [], []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            mapped_label = self.category_mapping.get(int(ann["category_id"]))
            if mapped_label is None or w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(mapped_label)
            areas.append(float(ann.get("area", w * h)))

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        return image_tensor, target


def _collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _to_device(items, device):
    if isinstance(items, torch.Tensor):
        return items.to(device)
    if isinstance(items, list):
        return [_to_device(item, device) for item in items]
    if isinstance(items, dict):
        return {k: _to_device(v, device) for k, v in items.items()}
    return items


def runSSDLite(fold: str, fold_dir: str, root_data_dir: str | Path) -> None:
    dataset_config = CriarLabelsSSDLite(fold, root_data_dir)
    cfg = get_config()

    num_classes = len(dataset_config.class_names) + 1  # +1 for background
    model = _build_model(num_classes, cfg)

    device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)

    train_dataset = _CocoDataset(dataset_config.train, dataset_config.category_mapping)
    val_dataset = _CocoDataset(dataset_config.val, dataset_config.category_mapping)
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

    print(
        f"[SSDLite] Iniciando treino | device={device} | epochs={cfg.epochs} "
        f"| batch={cfg.batch_size} | lr={cfg.learning_rate} "
        f"| classes={len(dataset_config.class_names)}",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = cfg.optimizer.strip().lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
    )

    best_state = None
    best_loss = float("inf")

    for epoch in range(cfg.epochs):
        print(f"[SSDLite] Época {epoch + 1}/{cfg.epochs} - treinamento", flush=True)
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
                bbox_loss = loss_dict.get("bbox_regression", torch.tensor(0.0)).item()
                cls_loss = loss_dict.get("classification", torch.tensor(0.0)).item()
                print(
                    f"[SSDLite]   batch {batch_idx}/{len(train_loader)} "
                    f"loss={losses.item():.4f} "
                    f"(cls={cls_loss:.4f}, bbox={bbox_loss:.4f})",
                    flush=True,
                )

        val_loss = 0.0
        val_batches = 0
        print(f"[SSDLite] Época {epoch + 1}/{cfg.epochs} - validação", flush=True)
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader, start=1):
                images = _to_device(images, device)
                targets = _to_device(targets, device)
                # SSD only returns losses in train mode
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
                val_loss += losses.item()
                val_batches += 1
                if batch_idx == 1 or batch_idx % 10 == 0:
                    print(
                        f"[SSDLite]   val batch {batch_idx}/{len(val_loader)} loss={losses.item():.4f}",
                        flush=True,
                    )

        avg_train_loss = running_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, val_batches)
        print(
            f"[SSDLite] Época {epoch + 1}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}",
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

    target_dir = Path(fold_dir) / "SSDLite"
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, target_dir / "best.pth")
    print(f"[SSDLite] Treinamento concluído | melhor val_loss={best_loss:.4f}", flush=True)
