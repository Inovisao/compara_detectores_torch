from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from backbones._common import AUX_CHANNELS, dw_sep_conv

# Mapeamento espacial para entrada 320×320:
#   stem (conv1 stride=2 + maxpool stride=2) → 80×80×64
#   layer1 (sem stride)                      → 80×80×64
#   layer2 (stride=2)                        → 40×40×128
#   layer3 (stride=2)  ← f1                 → 20×20×256
#   layer4 (stride=2)  ← f2                 → 10×10×512
#   aux ×4                                   → 5×5, 3×3, 2×2, 1×1


class ResNet18SSDBackbone(nn.Module):
    """
    Backbone ResNet-18 com pirâmide auxiliar compatível com SSDLiteHead.
    Produz 6 feature maps (20×20 → 1×1) para entrada 320×320.
    """

    out_channels: List[int] = [256, 512] + AUX_CHANNELS

    def __init__(self) -> None:
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3   # → 20×20×256
        self.layer4 = net.layer4   # → 10×10×512

        prev_ch = 512
        aux = []
        for out_ch in AUX_CHANNELS:
            aux.append(dw_sep_conv(prev_ch, out_ch))
            prev_ch = out_ch
        self.aux_layers = nn.ModuleList(aux)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f1 = self.layer3(x)   # 20×20×256
        f2 = self.layer4(f1)  # 10×10×512

        features = [f1, f2]
        feat = f2
        for layer in self.aux_layers:
            feat = layer(feat)
            features.append(feat)
        return OrderedDict((str(i), f) for i, f in enumerate(features))


__all__ = ["ResNet18SSDBackbone"]
