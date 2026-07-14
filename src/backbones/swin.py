from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t

from backbones._common import AUX_CHANNELS, dw_sep_conv


class SwinTinySSDLiteBackbone(nn.Module):
    """Swin Transformer Tiny ImageNet com saída NCHW para a cabeça SSDLite."""

    out_channels: List[int] = [384, 768] + AUX_CHANNELS

    def __init__(self, weights: Swin_T_Weights | None = Swin_T_Weights.IMAGENET1K_V1) -> None:
        super().__init__()
        base = swin_t(weights=weights)
        # O Swin do torchvision mantém os tokens como NHWC dentro de features.
        self.stage1 = base.features[:6]  # terceiro estágio: 20x20x384
        self.stage2 = base.features[6:]
        self.norm = base.norm
        self.to_nchw = base.permute

        channels = 768
        self.aux_layers = nn.ModuleList()
        for out_channels in AUX_CHANNELS:
            self.aux_layers.append(dw_sep_conv(channels, out_channels))
            channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stage3_tokens = self.stage1(x)
        f1 = self.to_nchw(stage3_tokens)
        f2 = self.to_nchw(self.norm(self.stage2(stage3_tokens)))
        features = [f1, f2]
        for layer in self.aux_layers:
            f2 = layer(f2)
            features.append(f2)
        return OrderedDict((str(index), feature) for index, feature in enumerate(features))


__all__ = ["SwinTinySSDLiteBackbone"]
