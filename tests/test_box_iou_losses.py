from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from losses import box_iou_loss, normalize_box_loss


@pytest.mark.parametrize("loss_name", ["ciou", "inner_mpdiou", "wise_iou", "siou"])
def test_box_losses_are_zero_for_identical_boxes(loss_name):
    boxes = torch.tensor([[4.0, 5.0, 16.0, 20.0]], requires_grad=True)
    loss = box_iou_loss(boxes, boxes, loss_name)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert torch.isfinite(boxes.grad).all()


@pytest.mark.parametrize("loss_name", ["ciou", "inner_mpdiou", "wise_iou", "siou"])
def test_box_losses_penalize_misaligned_boxes(loss_name):
    prediction = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    assert box_iou_loss(prediction, target, loss_name).item() > 0


def test_loss_aliases_are_normalized():
    assert normalize_box_loss("Inner-MPDIoU") == "inner_mpdiou"
    assert normalize_box_loss("Wise IoU") == "wise_iou"
