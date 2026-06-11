from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Dict, List

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from Detectors.ViT.GeraLabels import CriarLabelsViT, SplitPaths, ViTDatasetConfig
from Detectors.ViT.config import get_config


class _CocoDataset(Dataset):
    def __init__(self, split: SplitPaths, category_mapping: Dict[int, int]) -> None:
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

        image_path = self.images_dir / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found for ViT training: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations: List[Dict] = []
        for ann in self.annotations_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            mapped_label = self.category_mapping.get(int(ann["category_id"]))
            if mapped_label is None or w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "image_id": image_id,
                    "category_id": mapped_label,
                    "bbox": [x, y, w, h],
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        return image, {"image_id": image_id, "annotations": annotations}


def _collate_fn(batch, image_processor):
    images, targets = zip(*batch)
    encoding = image_processor(images=list(images), annotations=list(targets), return_tensors="pt")
    return encoding["pixel_values"], encoding["labels"]


def _prepare_dataloaders(dataset_config: ViTDatasetConfig, image_processor):
    train_dataset = _CocoDataset(dataset_config.train, dataset_config.category_mapping)
    val_dataset = _CocoDataset(dataset_config.val, dataset_config.category_mapping)

    cfg = get_config()
    print(
        f"[ViT] Preparando dataloaders - train={len(train_dataset)} imgs ({dataset_config.train.images}), "
        f"val={len(val_dataset)} imgs ({dataset_config.val.images}), classes={len(dataset_config.class_names)}",
        flush=True,
    )

    collate_fn = partial(_collate_fn, image_processor=image_processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
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


def runViT(fold: str, fold_dir: str, root_data_dir: str | Path) -> None:
    dataset_config = CriarLabelsViT(fold, root_data_dir)
    cfg = get_config()

    id2label = {idx: name for idx, name in enumerate(dataset_config.class_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    image_processor = AutoImageProcessor.from_pretrained(
        cfg.model_name, size={"height": cfg.image_size, "width": cfg.image_size}
    )
    model = AutoModelForObjectDetection.from_pretrained(
        cfg.model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_loader, val_loader, hyperparams = _prepare_dataloaders(dataset_config, image_processor)

    device_str = hyperparams.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)
    print(
        f"[ViT] Iniciando treino | model={hyperparams.model_name} | device={device} | epochs={hyperparams.epochs} "
        f"| batch={hyperparams.batch_size} | lr={hyperparams.learning_rate}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hyperparams.lr_step_size,
        gamma=hyperparams.lr_gamma,
    )

    def _checkpoint_state() -> dict:
        return {
            "model_state": model.state_dict(),
            "model_name": hyperparams.model_name,
            "image_size": hyperparams.image_size,
            "class_names": dataset_config.class_names,
            "category_mapping": dataset_config.category_mapping,
        }

    best_state = None
    best_loss = float("inf")

    for epoch in range(hyperparams.epochs):
        print(f"[ViT] Época {epoch + 1}/{hyperparams.epochs} - treinamento", flush=True)
        model.train()
        running_loss = 0.0
        for batch_idx, (pixel_values, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device)
            labels = _to_device(labels, device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(
                    f"[ViT]   batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        print(f"[ViT] Época {epoch + 1}/{hyperparams.epochs} - validação", flush=True)
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (pixel_values, labels) in enumerate(val_loader, start=1):
                pixel_values = pixel_values.to(device)
                labels = _to_device(labels, device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
                val_batches += 1
                if batch_idx == 1 or batch_idx % 10 == 0:
                    print(
                        f"[ViT]   val batch {batch_idx}/{len(val_loader)} loss={outputs.loss.item():.4f}",
                        flush=True,
                    )

        avg_val_loss = val_loss / max(1, val_batches)
        print(
            f"[ViT] Época {epoch + 1}: train_loss={running_loss / max(1, len(train_loader)):.4f} "
            f"| val_loss={avg_val_loss:.4f}",
            flush=True,
        )
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = _checkpoint_state()

        lr_scheduler.step()

    if best_state is None:
        best_state = _checkpoint_state()

    target_dir = Path(fold_dir) / "ViT"
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, target_dir / "best.pth")
    print(f"[ViT] Treinamento concluído | melhor val_loss={best_loss:.4f}", flush=True)
