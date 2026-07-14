"""Losses IoU alinhadas para caixas no formato ``(x1, y1, x2, y2)``."""

from __future__ import annotations

import math

import torch
from torch import Tensor


BOX_LOSSES = ("ciou", "inner_mpdiou", "wise_iou", "siou")


def normalize_box_loss(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ciou": "ciou",
        "complete_iou": "ciou",
        "inner_mpdiou": "inner_mpdiou",
        "innermpdiou": "inner_mpdiou",
        "wise_iou": "wise_iou",
        "wiou": "wise_iou",
        "siou": "siou",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Box loss não suportada: {name}. Use: {', '.join(BOX_LOSSES)}") from exc


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean() if loss.numel() else loss.sum()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction deve ser 'none', 'mean' ou 'sum'.")


def _geometry(boxes1: Tensor, boxes2: Tensor, eps: float) -> tuple[Tensor, ...]:
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError("As caixas devem ter o mesmo formato [N, 4] em xyxy.")
    x1, y1, x2, y2 = boxes1.unbind(-1)
    gx1, gy1, gx2, gy2 = boxes2.unbind(-1)
    w, h = (x2 - x1).clamp_min(eps), (y2 - y1).clamp_min(eps)
    gw, gh = (gx2 - gx1).clamp_min(eps), (gy2 - gy1).clamp_min(eps)
    inter = (torch.minimum(x2, gx2) - torch.maximum(x1, gx1)).clamp_min(0) * (
        torch.minimum(y2, gy2) - torch.maximum(y1, gy1)
    ).clamp_min(0)
    union = w * h + gw * gh - inter
    iou = inter / union.clamp_min(eps)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    cw = torch.maximum(x2, gx2) - torch.minimum(x1, gx1)
    ch = torch.maximum(y2, gy2) - torch.minimum(y1, gy1)
    return iou, w, h, gw, gh, cx, cy, gcx, gcy, cw, ch


def ciou_loss(boxes1: Tensor, boxes2: Tensor, reduction: str = "mean", eps: float = 1e-7) -> Tensor:
    """Complete-IoU, usada como baseline."""
    iou, w, h, gw, gh, cx, cy, gcx, gcy, cw, ch = _geometry(boxes1, boxes2, eps)
    rho2 = (cx - gcx).square() + (cy - gcy).square()
    c2 = cw.square() + ch.square() + eps
    v = 4 / math.pi**2 * (torch.atan(gw / gh) - torch.atan(w / h)).square()
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return _reduce(1 - iou + rho2 / c2 + alpha * v, reduction)


def inner_mpdiou_loss(
    boxes1: Tensor,
    boxes2: Tensor,
    reduction: str = "mean",
    eps: float = 1e-7,
    inner_ratio: float = 0.7,
) -> Tensor:
    """Inner-MPDIoU com IoU das caixas internas e penalidade dos dois cantos."""
    if inner_ratio <= 0:
        raise ValueError("inner_ratio deve ser maior que zero.")
    iou, w, h, gw, gh, cx, cy, gcx, gcy, cw, ch = _geometry(boxes1, boxes2, eps)
    inner_pred = torch.stack((cx - w * inner_ratio / 2, cy - h * inner_ratio / 2,
                              cx + w * inner_ratio / 2, cy + h * inner_ratio / 2), dim=-1)
    inner_target = torch.stack((gcx - gw * inner_ratio / 2, gcy - gh * inner_ratio / 2,
                                gcx + gw * inner_ratio / 2, gcy + gh * inner_ratio / 2), dim=-1)
    inner_iou = _geometry(inner_pred, inner_target, eps)[0]
    d1 = (boxes1[:, :2] - boxes2[:, :2]).square().sum(dim=-1)
    d2 = (boxes1[:, 2:] - boxes2[:, 2:]).square().sum(dim=-1)
    # O termo externo estabiliza a normalização quando as caixas não se cruzam.
    corner_penalty = (d1 + d2) / (cw.square() + ch.square() + eps)
    return _reduce(1 - inner_iou + corner_penalty, reduction)


def wise_iou_loss(boxes1: Tensor, boxes2: Tensor, reduction: str = "mean", eps: float = 1e-7) -> Tensor:
    """Wise-IoU v1: CIoU-free com atenção dinâmica à distância do centro."""
    iou, _, _, _, _, cx, cy, gcx, gcy, cw, ch = _geometry(boxes1, boxes2, eps)
    distance = ((cx - gcx).square() + (cy - gcy).square()) / (cw.square() + ch.square() + eps)
    # A distância é um fator de foco, não um segundo caminho de gradiente.
    distance_attention = torch.exp(distance.detach())
    return _reduce((1 - iou) * distance_attention, reduction)


def siou_loss(boxes1: Tensor, boxes2: Tensor, reduction: str = "mean", eps: float = 1e-7) -> Tensor:
    """Scylla-IoU (SIoU), com custos de ângulo, distância e forma."""
    iou, w, h, gw, gh, cx, cy, gcx, gcy, cw, ch = _geometry(boxes1, boxes2, eps)
    dx, dy = gcx - cx, gcy - cy
    sigma = torch.sqrt(dx.square() + dy.square() + eps)
    sin_alpha = torch.minimum(dx.abs(), dy.abs()) / sigma
    angle_cost = torch.cos(torch.asin(sin_alpha.clamp(max=1 - eps)) * 2 - math.pi / 2)
    gamma = angle_cost - 2
    rho_x, rho_y = (dx / (cw + eps)).square(), (dy / (ch + eps)).square()
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
    omega_w = (w - gw).abs() / torch.maximum(w, gw)
    omega_h = (h - gh).abs() / torch.maximum(h, gh)
    shape_cost = (1 - torch.exp(-omega_w)).pow(4) + (1 - torch.exp(-omega_h)).pow(4)
    return _reduce(1 - iou + (distance_cost + shape_cost) / 2, reduction)


def box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, loss_type: str = "ciou", reduction: str = "mean", **kwargs
) -> Tensor:
    """Despacha uma das losses disponíveis para pares de caixas correspondentes."""
    name = normalize_box_loss(loss_type)
    if name == "ciou":
        return ciou_loss(boxes1, boxes2, reduction=reduction)
    if name == "inner_mpdiou":
        return inner_mpdiou_loss(boxes1, boxes2, reduction=reduction, **kwargs)
    if name == "wise_iou":
        return wise_iou_loss(boxes1, boxes2, reduction=reduction)
    return siou_loss(boxes1, boxes2, reduction=reduction)


__all__ = ["BOX_LOSSES", "box_iou_loss", "ciou_loss", "inner_mpdiou_loss", "normalize_box_loss", "siou_loss", "wise_iou_loss"]
