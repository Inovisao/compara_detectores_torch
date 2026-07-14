"""Integra as losses IoU personalizadas aos detectores do torchvision."""

from __future__ import annotations

from types import MethodType
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.box_iou import box_iou_loss, normalize_box_loss


def configure_ssd_box_loss(model, loss_type: str, *, inner_ratio: float = 0.7) -> None:
    """Substitui apenas a regressão de caixas do SSD/SSDLite.

    A classificação, matching de âncoras e hard-negative mining permanecem os
    da implementação oficial do torchvision.
    """
    loss_type = normalize_box_loss(loss_type)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs) -> Dict[str, Tensor]:
        bbox_regression, cls_logits = head_outputs["bbox_regression"], head_outputs["cls_logits"]
        num_foreground = 0
        bbox_losses: List[Tensor] = []
        cls_targets: List[Tensor] = []
        for target, regression, logits, anchors_image, matched in zip(
            targets, bbox_regression, cls_logits, anchors, matched_idxs
        ):
            foreground = torch.where(matched >= 0)[0]
            matched_gt = matched[foreground]
            num_foreground += matched_gt.numel()
            target_boxes = target["boxes"][matched_gt]
            predicted_boxes = self.box_coder.decode_single(regression[foreground], anchors_image[foreground])
            bbox_losses.append(
                box_iou_loss(predicted_boxes, target_boxes, loss_type, reduction="sum", inner_ratio=inner_ratio)
            )
            classes = torch.zeros((logits.size(0),), dtype=target["labels"].dtype, device=logits.device)
            classes[foreground] = target["labels"][matched_gt]
            cls_targets.append(classes)

        bbox_loss = torch.stack(bbox_losses).sum()
        cls_targets_tensor = torch.stack(cls_targets)
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets_tensor.view(-1), reduction="none").view(
            cls_targets_tensor.size()
        )
        foreground = cls_targets_tensor > 0
        num_negative = self.neg_to_pos_ratio * foreground.sum(1, keepdim=True)
        negative_loss = cls_loss.clone()
        negative_loss[foreground] = -float("inf")
        negative_indices = negative_loss.sort(1, descending=True)[1].sort(1)[1] < num_negative
        normalizer = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss / normalizer,
            "classification": (cls_loss[foreground].sum() + cls_loss[negative_indices].sum()) / normalizer,
        }

    model.compute_loss = MethodType(compute_loss, model)


def configure_retinanet_box_loss(model, loss_type: str, *, inner_ratio: float = 0.7) -> None:
    """Substitui a regressão do RetinaNet sem alterar classificação ou matching."""
    loss_type = normalize_box_loss(loss_type)
    # CIoU já é suportada nativamente pelo torchvision e não precisa de patch.
    if loss_type == "ciou":
        model.head.regression_head._loss_type = "ciou"
        return

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs) -> Tensor:
        losses: List[Tensor] = []
        for target, regression, anchors_image, matched in zip(
            targets, head_outputs["bbox_regression"], anchors, matched_idxs
        ):
            foreground = torch.where(matched >= 0)[0]
            target_boxes = target["boxes"][matched[foreground]]
            predicted_boxes = self.box_coder.decode_single(regression[foreground], anchors_image[foreground])
            losses.append(
                box_iou_loss(predicted_boxes, target_boxes, loss_type, reduction="sum", inner_ratio=inner_ratio)
                / max(1, foreground.numel())
            )
        return torch.stack(losses).sum() / max(1, len(targets))

    model.head.regression_head.compute_loss = MethodType(compute_loss, model.head.regression_head)


__all__ = ["configure_retinanet_box_loss", "configure_ssd_box_loss"]
