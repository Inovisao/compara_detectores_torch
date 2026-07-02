from __future__ import annotations

import torchvision.models.detection as detection_models
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn_model(backbone: str, num_classes: int):
    key = backbone.strip().lower()
    if key == "resnet50_fpn":
        model = detection_models.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif key == "resnet50_fpn_v2":
        builder = getattr(detection_models, "fasterrcnn_resnet50_fpn_v2", None)
        if builder is None:
            raise ValueError(
                "fasterrcnn_resnet50_fpn_v2 não está disponível nesta versão do torchvision."
            )
        model = builder(weights="DEFAULT")
    elif key == "mobilenet_v3_large_fpn":
        builder = getattr(detection_models, "fasterrcnn_mobilenet_v3_large_fpn", None)
        if builder is None:
            raise ValueError(
                "fasterrcnn_mobilenet_v3_large_fpn não está disponível nesta versão do torchvision."
            )
        model = builder(weights="DEFAULT")
    else:
        raise ValueError(
            "Backbone FasterRCNN não suportado: "
            f"{backbone}. Use resnet50_fpn, resnet50_fpn_v2 ou mobilenet_v3_large_fpn."
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_retinanet_model(backbone: str, num_classes: int):
    key = backbone.strip().lower()
    backbone_weights = ResNet50_Weights.IMAGENET1K_V2
    if key == "resnet50_fpn":
        return retinanet_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
            weights_backbone=backbone_weights,
        )
    if key == "resnet50_fpn_v2":
        builder = getattr(detection_models, "retinanet_resnet50_fpn_v2", None)
        if builder is None:
            raise ValueError(
                "retinanet_resnet50_fpn_v2 não está disponível nesta versão do torchvision."
            )
        return builder(
            weights=None,
            num_classes=num_classes,
            weights_backbone=backbone_weights,
        )
    raise ValueError(
        "Backbone RetinaNet não suportado: "
        f"{backbone}. Use resnet50_fpn ou resnet50_fpn_v2."
    )


__all__ = ["build_fasterrcnn_model", "build_retinanet_model"]
