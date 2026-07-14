from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn

from backbones._common import AUX_CHANNELS, dw_sep_conv


class _ConvBNAct(nn.Sequential):
    """Convolução usada pelos blocos GELAN (Conv + BN + SiLU)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class _RepConv(nn.Module):
    """Versão de treino do RepConv usado pelo GELAN/YOLOv9.

    Os dois ramos podem ser reparametrizados para uma única convolução na
    inferência, mas mantê-los explícitos torna o backbone simples e estável
    para fine-tuning dentro deste projeto.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv3(x) + self.conv1(x))


class _RepNCSPELAN(nn.Module):
    """Bloco ELAN com caminhos de gradiente curtos, inspirado no GELAN."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.expand = _ConvBNAct(in_channels, hidden_channels * 2, kernel_size=1)
        self.block1 = nn.Sequential(_RepConv(hidden_channels), _ConvBNAct(hidden_channels, hidden_channels))
        self.block2 = nn.Sequential(_RepConv(hidden_channels), _ConvBNAct(hidden_channels, hidden_channels))
        self.project = _ConvBNAct(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.expand(x).chunk(2, dim=1)
        third = self.block1(second)
        fourth = self.block2(third)
        return self.project(torch.cat((first, second, third, fourth), dim=1))


class GELANSSDLiteBackbone(nn.Module):
    """GELAN compacto para SSDLite, treinado do zero.

    Produz seis mapas de características, de 20x20 até 1x1 para uma entrada
    320x320. GELAN não possui pesos ImageNet oficiais no torchvision, portanto
    este backbone é inicializado do zero.
    """

    out_channels: List[int] = [256, 512] + AUX_CHANNELS

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            _ConvBNAct(3, 64, stride=2),       # 160x160
            _ConvBNAct(64, 128, stride=2),     # 80x80
            _RepNCSPELAN(128, 128, 64),
            _ConvBNAct(128, 256, stride=2),    # 40x40
            _RepNCSPELAN(256, 256, 128),
            _ConvBNAct(256, 256, stride=2),    # 20x20
            _RepNCSPELAN(256, 256, 128),
        )
        self.stage2 = nn.Sequential(
            _ConvBNAct(256, 512, stride=2),    # 10x10
            _RepNCSPELAN(512, 512, 256),
        )

        channels = 512
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


__all__ = ["GELANSSDLiteBackbone"]
