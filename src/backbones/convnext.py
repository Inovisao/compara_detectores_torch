from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from backbones._common import AUX_CHANNELS, dw_sep_conv


class ConvNeXtTinySSDLiteBackbone(nn.Module):
    """ConvNeXt-Tiny ImageNet com pirâmide auxiliar compatível com SSDLite."""

    out_channels: List[int] = [384, 768] + AUX_CHANNELS

    def __init__(self, weights: ConvNeXt_Tiny_Weights | None = ConvNeXt_Tiny_Weights.IMAGENET1K_V1) -> None:
        super().__init__()
        base = convnext_tiny(weights=weights)
        # features[0:6] termina no terceiro estágio (20x20 para entrada 320).
        self.stage1 = base.features[:6]
        self.stage2 = base.features[6:]

        channels = 768
        self.aux_layers = nn.ModuleList()
        for out_channels in AUX_CHANNELS:
            self.aux_layers.append(dw_sep_conv(channels, out_channels))
            channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        features = [f1, f2]
        for layer in self.aux_layers:
            f2 = layer(f2)
            features.append(f2)
        return OrderedDict((str(index), feature) for index, feature in enumerate(features))


__all__ = ["ConvNeXtTinySSDLiteBackbone"]
