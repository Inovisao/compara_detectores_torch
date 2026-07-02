from __future__ import annotations

from backbones.mobilenetv2 import MobileNetV2SSDLiteBackbone
from backbones.resnet import ResNet18SSDBackbone


SSDLITE_BACKBONES = {
    "mobilenetv2": MobileNetV2SSDLiteBackbone,
    "mobilenet_v2": MobileNetV2SSDLiteBackbone,
    "resnet18": ResNet18SSDBackbone,
    "resnet_18": ResNet18SSDBackbone,
}


def build_ssdlite_backbone(name: str):
    key = name.strip().lower()
    try:
        return SSDLITE_BACKBONES[key]()
    except KeyError as exc:
        supported = ", ".join(sorted(SSDLITE_BACKBONES))
        raise ValueError(
            f"Backbone SSDLite não suportado: {name}. Use um destes: {supported}"
        ) from exc


__all__ = ["SSDLITE_BACKBONES", "build_ssdlite_backbone"]
