"""Evaluation: COCO mAP + custom counting/classification metrics."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

logger = logging.getLogger(__name__)


def _calculate_iou(box1_xywh, box2_xywh):
    x1, y1, w1, h1 = box1_xywh
    x2, y2, w2, h2 = box2_xywh
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    ix = max(0, min(x1_max, x2_max) - max(x1, x2))
    iy = max(0, min(y1_max, y2_max) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def compute_coco_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    metric = MeanAveragePrecision()
    metric.update(predictions, ground_truths)
    result = metric.compute()
    return {
        "mAP": result["map"].item(),
        "mAP50": result["map_50"].item(),
        "mAP75": result["map_75"].item(),
    }


def compute_counting_metrics(pred_counts, gt_counts) -> dict:
    return {
        "MAE": MeanAbsoluteError()(pred_counts, gt_counts).item(),
        "RMSE": MeanSquaredError(squared=False)(pred_counts, gt_counts).item(),
        "pearson_r": PearsonCorrCoef()(pred_counts.float(), gt_counts.float()).item(),
    }


def classify_predictions(
    predictions_per_image: dict[str, list],
    ground_truth_per_image: dict[str, list],
    iou_threshold: float,
    classes: dict,
) -> tuple[list[int], list[int], int, int]:
    """Match predictions to ground truth per image. Returns (pred_labels, gt_labels, total_tp, total_fp)."""
    pred_labels = []
    gt_labels = []
    total_tp = 0
    total_fp = 0

    for img_name in predictions_per_image:
        gt_bboxes = ground_truth_per_image.get(img_name, [])
        pred_bboxes = predictions_per_image[img_name]
        matched_gt = set()

        for pred in pred_bboxes:
            best_iou = 0.0
            best_gt_idx = None
            for i, gt in enumerate(gt_bboxes):
                iou = _calculate_iou(pred[:4], gt[:4])
                if iou >= iou_threshold and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt_idx = i
            if best_gt_idx is not None:
                matched_gt.add(best_gt_idx)
                gt_class = gt_bboxes[best_gt_idx][4]
                pred_class = int(pred[4])
                pred_labels.append(pred_class)
                gt_labels.append(gt_class)
                if gt_class == pred_class:
                    total_tp += 1
                else:
                    total_fp += 1
            else:
                pred_labels.append(int(pred[4]))
                gt_labels.append(0)
                total_fp += 1

        for i, gt in enumerate(gt_bboxes):
            if i not in matched_gt:
                gt_labels.append(gt[4])
                pred_labels.append(0)

    return pred_labels, gt_labels, total_tp, total_fp


def compute_classification_metrics(pred_labels, gt_labels, num_classes) -> dict:
    preds = torch.tensor(pred_labels)
    targets = torch.tensor(gt_labels)
    return {
        "precision": BinaryPrecision()(preds, targets).item(),
        "recall": BinaryRecall()(preds, targets).item(),
        "f1": BinaryF1Score()(preds, targets).item(),
    }


def evaluate_detector(detector_instance, test_loader, classes, iou_threshold=0.2) -> dict:
    """Run full evaluation: predict all images, compute all metrics."""
    predictions_map = []
    ground_truths_map = []
    pred_per_image = {}
    gt_per_image = {}
    pred_counts = []
    gt_counts = []

    for images, targets in test_loader:
        outputs = detector_instance.predict(list(images))

        for img_idx, target in enumerate(targets):
            img_id = target["image_id"].item()
            gt_boxes = target["boxes"].tolist()
            gt_labels = target["labels"].tolist()

            output = outputs[img_idx]
            pred_boxes = output["boxes"].tolist()
            pred_scores = output["scores"].tolist()
            pred_labels_out = output["labels"].tolist()

            pred_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in pred_boxes]
            gt_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in gt_boxes]

            pred_per_image[str(img_id)] = [
                pred_boxes_xywh[i] + [pred_labels_out[i], pred_scores[i]]
                for i in range(len(pred_boxes))
            ]
            gt_per_image[str(img_id)] = [
                gt_boxes_xywh[i] + [gt_labels[i]] for i in range(len(gt_boxes))
            ]
            pred_counts.append(len(pred_boxes))
            gt_counts.append(len(gt_boxes))

            predictions_map.append({
                "boxes": output["boxes"],
                "scores": output["scores"],
                "labels": output["labels"],
            })
            ground_truths_map.append({
                "boxes": target["boxes"],
                "labels": target["labels"],
            })

    coco = compute_coco_metrics(predictions_map, ground_truths_map)
    counting = compute_counting_metrics(
        torch.tensor(pred_counts), torch.tensor(gt_counts)
    )
    pred_lbl, gt_lbl, tp, fp = classify_predictions(
        pred_per_image, gt_per_image, iou_threshold, classes
    )
    class_metrics = compute_classification_metrics(pred_lbl, gt_lbl, len(classes))

    return {
        **coco, **counting, **class_metrics,
        "TP": tp, "FP": fp, "total_preds": len(pred_lbl),
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2))


def append_csv_row(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
