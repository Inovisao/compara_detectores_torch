from __future__ import annotations

from functools import partial

import torch.nn as nn

AUX_CHANNELS = [256, 256, 256, 64]

# GroupNorm com 32 grupos é estável em qualquer tamanho espacial (inclusive 1×1)
# e funciona para todos os channel counts usados aqui (64, 96, 256, 320, 512).
NORM_LAYER = partial(nn.GroupNorm, 32)


def dw_sep_conv(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        NORM_LAYER(in_ch),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        NORM_LAYER(out_ch),
        nn.ReLU6(inplace=True),
    )


__all__ = ["AUX_CHANNELS", "NORM_LAYER", "dw_sep_conv"]
