"""DETR detector using torchvision."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision

from detectors.base import Detector
from engine.trainer import train_torchvision_model


class DETRDetector(Detector):
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def architectures(cls) -> list[str]:
        return ["detr_resnet50", "detr_resnet101"]

    @classmethod
    def default_hparams(cls) -> dict:
        return {
            "lr": {"default": 0.0001, "help": "(float) initial learning rate (backbone lr = lr/10)"},
            "epochs": {"default": 150, "help": "(int)"},
            "batch_size": {"default": 8, "help": "(int)"},
            "optimizer": {"default": "AdamW", "help": "AdamW | Adam"},
            "weight_decay": {"default": 0.0001, "help": "(float)"},
            "scheduler": {"default": "step", "help": "step | cosine | none"},
            "lr_drop": {"default": 100, "help": "(int) epoch to drop LR by 10x (DETR convention)"},
            "patience": {"default": 20, "help": "(int) early stopping, 0 = disabled"},
        }

    def _build_model(self, arch: str, num_classes: int) -> torch.nn.Module:
        nc = num_classes + 1  # torchvision includes no-object class
        if arch == "detr_resnet50":
            model = torchvision.models.detection.detr_resnet50(
                weights="DEFAULT", num_classes=nc
            )
        elif arch == "detr_resnet101":
            model = torchvision.models.detection.detr_resnet50(
                weights=None, num_classes=nc
            )
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                "resnet101", weights=None
            )
            model.backbone = backbone
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        return model

    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        model = self._build_model(config["architecture"], config["num_classes"])
        self.model = model

        config = dict(config)
        if config.get("scheduler") == "step":
            config["step_size"] = config.pop("lr_drop", 100)

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
