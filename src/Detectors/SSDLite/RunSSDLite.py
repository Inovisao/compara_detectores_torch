from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead

from backbones import MobileNetV2SSDLiteBackbone
from backbones._common import NORM_LAYER
from Detectors.SSDLite.GeraLabels import (
    CriarLabelsSSDLite,
    SSDLiteDatasetConfig,
    SplitPaths,
)
from Detectors.SSDLite.config import get_config

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transforms(train: bool) -> A.Compose:
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,
    )
    _pad = dict(min_height=320, min_width=320, border_mode=0, value=0, position="center")
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=320),
            A.PadIfNeeded(**_pad),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=10, border_mode=0, p=0.5,
            ),
            A.OneOf([
                A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
                A.HueSaturationValue(10, 30, 20, p=1.0),
                A.CLAHE(p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 30), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=320),
            A.PadIfNeeded(**_pad),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ], bbox_params=bbox_params)


def _build_model(num_classes: int, cfg) -> SSD:
    backbone = MobileNetV2SSDLiteBackbone()

    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    head = SSDLiteHead(
        in_channels=backbone.out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=NORM_LAYER,
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


def _freeze_backbone(model: SSD) -> None:
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad = False
    print("[SSDLite] Backbone congelado", flush=True)


def _unfreeze_backbone(model: SSD, optimizer, epoch: int, lr_reduced: float) -> None:
    """
    Época 10: descongela stage2 + aux_layers (camadas mais específicas).
    Época 20: descongela stage1 (camadas mais genéricas), LR reduzido.
    Novos params são adicionados ao optimizer via add_param_group.
    """
    if epoch == 10:
        newly_unfrozen = []
        for name, param in model.named_parameters():
            if not param.requires_grad and (
                name.startswith("backbone.stage2")
                or name.startswith("backbone.aux_layers")
            ):
                param.requires_grad = True
                newly_unfrozen.append(param)
        if newly_unfrozen:
            optimizer.add_param_group({"params": newly_unfrozen, "lr": lr_reduced})
            print(
                f"[SSDLite] stage2 + aux_layers descongelados (época {epoch + 1}) "
                f"| LR={lr_reduced}",
                flush=True,
            )

    elif epoch == 20:
        newly_unfrozen = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                newly_unfrozen.append(param)
        if newly_unfrozen:
            optimizer.add_param_group({"params": newly_unfrozen, "lr": lr_reduced})
            print(
                f"[SSDLite] backbone totalmente descongelado (época {epoch + 1}) "
                f"| LR={lr_reduced}",
                flush=True,
            )


class _CocoDataset(Dataset):
    def __init__(
        self,
        split: SplitPaths,
        category_mapping: Dict[int, int],
        train: bool = False,
    ) -> None:
        self.images_dir = Path(split.images)
        self.category_mapping = category_mapping
        self.train = train
        self.transforms = _build_transforms(train)

        with open(split.annotations, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.images_info = sorted(data.get("images", []), key=lambda img: img["id"])
        self.annotations_by_image: Dict[int, List[Dict]] = {}
        for ann in data.get("annotations", []):
            self.annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    def __len__(self) -> int:
        return len(self.images_info)

    def __getitem__(self, idx: int):
        if idx == 0:
            print(f"[DEBUG] transform type: {type(self.transforms)}", flush=True)
            print(f"[DEBUG] train mode: {self.train}", flush=True)
        sample   = self.images_info[idx]
        image_id = int(sample["id"])

        image_path = self.images_dir / sample["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = self.annotations_by_image.get(image_id, [])
        img_h, img_w = image.shape[:2]
        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            mapped_label = self.category_mapping.get(int(ann["category_id"]))
            if mapped_label is None or w <= 0 or h <= 0:
                continue
            x2, y2 = x + w, y + h
            if x < 0 or y < 0 or x2 > img_w or y2 > img_h:
                x  = max(0.0, x)
                y  = max(0.0, y)
                x2 = min(float(img_w), x2)
                y2 = min(float(img_h), y2)
            if x2 - x < 2 or y2 - y < 2:
                continue
            boxes.append([x, y, x2, y2])
            labels.append(mapped_label)

        transformed = self.transforms(
            image=image,
            bboxes=boxes if boxes else [],
            labels=labels if labels else [],
        )
        image_tensor = transformed["image"]
        boxes  = list(transformed["bboxes"])
        labels = list(transformed["labels"])

        if boxes:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            target = {
                "boxes":    torch.tensor(boxes,  dtype=torch.float32),
                "labels":   torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area":     torch.tensor(areas,  dtype=torch.float32),
                "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
            }
        else:
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area":     torch.zeros((0,),   dtype=torch.float32),
                "iscrowd":  torch.zeros((0,),   dtype=torch.int64),
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

    log_path = Path(fold_dir) / "SSDLite" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("epoch,val_loss\n")
    log_file.flush()

    device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)

    train_dataset = _CocoDataset(
        dataset_config.train, dataset_config.category_mapping, train=True
    )
    val_dataset = _CocoDataset(
        dataset_config.val, dataset_config.category_mapping, train=False
    )
    _use_workers = cfg.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=_use_workers,
        prefetch_factor=2 if _use_workers else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=_use_workers,
        prefetch_factor=2 if _use_workers else None,
    )

    print(
        f"[SSDLite] Iniciando treino | device={device} | epochs={cfg.epochs} "
        f"| batch={cfg.batch_size} | lr={cfg.learning_rate} "
        f"| classes={len(dataset_config.class_names)}",
        flush=True,
    )

    train_empty = sum(
        1 for info in train_dataset.images_info
        if not train_dataset.annotations_by_image.get(int(info["id"]))
    )
    val_empty = sum(
        1 for info in val_dataset.images_info
        if not val_dataset.annotations_by_image.get(int(info["id"]))
    )
    print(
        f"[DEBUG] train={len(train_dataset)} ({train_empty} sem anotação, "
        f"{100 * train_empty / max(1, len(train_dataset)):.1f}%) | "
        f"val={len(val_dataset)} ({val_empty} sem anotação, "
        f"{100 * val_empty / max(1, len(val_dataset)):.1f}%)",
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

    _warmup_epochs = 3
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs - _warmup_epochs, eta_min=1e-6)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[_warmup_epochs])

    best_state = None
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        print(f"[SSDLite] Época {epoch + 1}/{cfg.epochs} - treinamento", flush=True)
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images  = _to_device(images, device)
            targets = _to_device(targets, device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += losses.item()
            if batch_idx == 1 or batch_idx % 10 == 0:
                bbox_loss = loss_dict.get("bbox_regression", torch.tensor(0.0)).item()
                cls_loss  = loss_dict.get("classification",  torch.tensor(0.0)).item()
                print(
                    f"[SSDLite]   batch {batch_idx}/{len(train_loader)} "
                    f"loss={losses.item():.4f} "
                    f"(cls={cls_loss:.4f}, bbox={bbox_loss:.4f})",
                    flush=True,
                )

        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[SSDLite] val  {epoch + 1}/{cfg.epochs}", leave=False)
            for images, targets in val_bar:
                images  = _to_device(images, device)
                targets = _to_device(targets, device)
                # SSD only returns losses in train mode
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
                val_loss += losses.item()
                val_batches += 1
                val_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))
        avg_val_loss   = val_loss / max(1, val_batches)
        print(
            f"[SSDLite] Época {epoch + 1}: train_loss={avg_train_loss:.4f} "
            f"| val_loss={avg_val_loss:.4f}",
            flush=True,
        )
        log_file.write(f"{epoch + 1},{avg_val_loss:.6f}\n")
        log_file.flush()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            best_state = {
                "model_state":      model.state_dict(),
                "num_classes":      num_classes,
                "class_names":      dataset_config.class_names,
                "category_mapping": dataset_config.category_mapping,
            }
        else:
            epochs_without_improvement += 1
            if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
                print(
                    f"[SSDLite] Early stopping na época {epoch + 1} "
                    f"({cfg.patience} épocas sem melhora).",
                    flush=True,
                )
                break

        lr_scheduler.step()

    if best_state is None:
        best_state = {
            "model_state":      model.state_dict(),
            "num_classes":      num_classes,
            "class_names":      dataset_config.class_names,
            "category_mapping": dataset_config.category_mapping,
        }

    target_dir = Path(fold_dir) / "SSDLite"
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, target_dir / "best.pth")
    log_file.close()
    print(f"[SSDLite] Treinamento concluído | melhor val_loss={best_loss:.4f}", flush=True)
