"""YOLOv8 detector using Ultralytics."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from detectors.base import Detector


class YOLOV8Detector(Detector):
    def __init__(self):
        self.model: YOLO | None = None

    @classmethod
    def architectures(cls) -> list[str]:
        return ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

    @classmethod
    def default_hparams(cls) -> dict:
        return {
            "lr": {"default": 0.001, "help": "(float) initial learning rate"},
            "epochs": {"default": 100, "help": "(int)"},
            "batch_size": {"default": 16, "help": "(int)"},
            "optimizer": {"default": "AdamW", "help": "AdamW | SGD | Adam"},
            "weight_decay": {"default": 0.0005, "help": "(float)"},
            "momentum": {"default": 0.937, "help": "(float) SGD momentum"},
            "scheduler": {"default": "cosine", "help": "cosine | linear | step | none"},
            "warmup_epochs": {"default": 3, "help": "(int)"},
            "patience": {"default": 10, "help": "(int) early stopping, 0 = disabled"},
            "imgsz": {"default": 640, "help": "(int) input image size"},
            "workers": {"default": 8, "help": "(int) dataloader workers"},
        }

    def _build_data_yaml(self, train_dir: str, val_dir: str, num_classes: int) -> str:
        data = {
            "path": ".",
            "train": train_dir,
            "val": val_dir,
            "names": {i: str(i) for i in range(num_classes)},
            "nc": num_classes,
        }
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="yolo_data_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        return path

    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        arch = config["architecture"]
        num_classes = config["num_classes"]
        img_size = config.get("imgsz", 640)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_img_dir = output_dir / "train_images"
        val_img_dir = output_dir / "val_images"
        train_label_dir = output_dir / "train_labels"
        val_label_dir = output_dir / "val_labels"
        for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for loader, img_dir, lbl_dir in [
            (train_loader, train_img_dir, train_label_dir),
            (val_loader, val_img_dir, val_label_dir),
        ]:
            for images, targets in loader:
                for img_tensor, target in zip(images, targets):
                    boxes = target["boxes"].tolist()
                    labels = target["labels"].tolist()
                    img_id = target["image_id"].item()
                    img_name = f"{img_id}.jpg"

                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(img_dir / img_name), img_np)

                    with open(lbl_dir / f"{img_id}.txt", "w") as f:
                        for box, label in zip(boxes, labels):
                            x1, y1, x2, y2 = box
                            cx = ((x1 + x2) / 2) / img_size
                            cy = ((y1 + y2) / 2) / img_size
                            bw = (x2 - x1) / img_size
                            bh = (y2 - y1) / img_size
                            f.write(f"{int(label) - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        data_yaml = self._build_data_yaml(
            str(train_img_dir), str(val_img_dir), num_classes
        )

        model = YOLO(f"{arch}.pt")
        model.train(
            data=data_yaml,
            epochs=config.get("epochs", 100),
            batch=config.get("batch_size", 16),
            imgsz=img_size,
            lr0=config.get("lr", 0.001),
            optimizer=config.get("optimizer", "AdamW"),
            weight_decay=config.get("weight_decay", 0.0005),
            momentum=config.get("momentum", 0.937),
            warmup_epochs=config.get("warmup_epochs", 3),
            patience=config.get("patience", 10) if config.get("patience", 0) > 0 else 0,
            cos_lr=config.get("scheduler") == "cosine",
            project=str(output_dir),
            name="train",
            exist_ok=True,
            verbose=False,
        )

        best_path = output_dir / "best.pt"
        src_best = output_dir / "train" / "weights" / "best.pt"
        if src_best.exists():
            shutil.copy(str(src_best), str(best_path))

        self.model = model
        return best_path

    def predict(self, images: list) -> list:
        results = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            output = self.model(img_np, verbose=False)[0]
            if output.boxes is not None:
                results.append({
                    "boxes": output.boxes.xyxy.cpu(),
                    "scores": output.boxes.conf.cpu(),
                    "labels": output.boxes.cls.cpu().long() + 1,
                })
            else:
                results.append({
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                })
        return results

    def load(self, path: Path) -> None:
        self.model = YOLO(str(path))
