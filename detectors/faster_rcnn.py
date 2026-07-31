"""Faster R-CNN detector using torchvision."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detectors.base import Detector
from engine.trainer import train_torchvision_model


class FasterRCNNDetector(Detector):
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def architectures(cls) -> list[str]:
        return ["resnet50", "resnet101"]

    @classmethod
    def default_hparams(cls) -> dict:
        return {
            "lr": {"default": 0.0001, "help": "(float) initial learning rate"},
            "epochs": {"default": 30, "help": "(int)"},
            "batch_size": {"default": 8, "help": "(int)"},
            "optimizer": {"default": "SGD", "help": "SGD | AdamW | Adam"},
            "weight_decay": {"default": 0.0005, "help": "(float)"},
            "momentum": {"default": 0.9, "help": "(float) SGD momentum"},
            "scheduler": {"default": "step", "help": "step | cosine | plateau | none"},
            "step_size": {"default": 10, "help": "(int) for step scheduler"},
            "gamma": {"default": 0.1, "help": "(float) LR decay factor"},
            "patience": {"default": 5, "help": "(int) early stopping, 0 = disabled"},
        }

    def _build_model(self, arch: str, num_classes: int) -> torch.nn.Module:
        nc = num_classes + 1  # torchvision includes background class 0
        if arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif arch == "resnet101":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                "resnet101", weights=None
            )
            model.backbone = backbone
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc)
        return model

    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        model = self._build_model(config["architecture"], config["num_classes"])
        self.model = model
        return train_torchvision_model(model, train_loader, val_loader, config, output_dir)

    def predict(self, images: list) -> list:
        self.model.eval()
        self.model.to(self.device)
        images = [img.to(self.device) for img in images]
        with torch.no_grad():
            outputs = self.model(images)
        results = []
        for output in outputs:
            results.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
        return results

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
