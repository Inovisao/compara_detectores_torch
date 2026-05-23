from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from backbones._common import AUX_CHANNELS, dw_sep_conv

# features[0:14]  → 20×20×96   (para entrada 320×320)
# features[14:18] → 10×10×320
_STAGE1_END = 14
_STAGE2_END = 18


class MobileNetV2SSDLiteBackbone(nn.Module):
    """
    Backbone MobileNetV2 com pirâmide auxiliar SSDLite.
    Produz 6 feature maps (20×20 → 1×1) para entrada 320×320.
    """

    out_channels: List[int] = [96, 320] + AUX_CHANNELS

    def __init__(self) -> None:
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.stage1 = base.features[:_STAGE1_END]
        self.stage2 = base.features[_STAGE1_END:_STAGE2_END]

        prev_ch = 320
        aux = []
        for out_ch in AUX_CHANNELS:
            aux.append(dw_sep_conv(prev_ch, out_ch))
            prev_ch = out_ch
        self.aux_layers = nn.ModuleList(aux)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        features = [f1, f2]
        feat = f2
        for layer in self.aux_layers:
            feat = layer(feat)
            features.append(feat)
        return OrderedDict((str(i), f) for i, f in enumerate(features))


__all__ = ["MobileNetV2SSDLiteBackbone"]
