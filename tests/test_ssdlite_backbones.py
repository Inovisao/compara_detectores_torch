from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from backbones.convnext import ConvNeXtTinySSDLiteBackbone
from backbones.factory import SSDLITE_BACKBONES, build_ssdlite_backbone
from backbones.gelan import GELANSSDLiteBackbone
from backbones.swin import SwinTinySSDLiteBackbone


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("gelan", GELANSSDLiteBackbone),
        ("convnext_tiny", ConvNeXtTinySSDLiteBackbone),
        ("swin_tiny", SwinTinySSDLiteBackbone),
    ],
)
def test_new_backbones_are_registered(name, expected_type):
    assert SSDLITE_BACKBONES[name] is expected_type


def test_unknown_backbone_reports_all_choices():
    with pytest.raises(ValueError, match="convnext_tiny"):
        build_ssdlite_backbone("not-a-backbone")


@pytest.mark.parametrize(
    "backbone",
    [
        GELANSSDLiteBackbone(),
        ConvNeXtTinySSDLiteBackbone(weights=None),
        SwinTinySSDLiteBackbone(weights=None),
    ],
)
def test_backbones_produce_ssdlite_feature_pyramid(backbone):
    backbone.eval()
    with torch.inference_mode():
        features = list(backbone(torch.zeros(1, 3, 320, 320)).values())

    assert len(features) == 6
    assert [feature.shape[1] for feature in features] == backbone.out_channels
    assert [tuple(feature.shape[-2:]) for feature in features] == [
        (20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)
    ]
