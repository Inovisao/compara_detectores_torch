from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from Detectors.Detr.detection.detr.model import DETRModel
from Detectors.Detr.utils.detection.detr.transforms import resize, infer_transforms
from Detectors.Detr.config import NUM_CLASSES, CLASSES, DEVICE


_model_cache: dict = {}


def _load_model(checkpoint_path: str):
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    model = DETRModel(num_classes=NUM_CLASSES, model="detr_resnet50")
    state = torch.load(checkpoint_path, map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    _model_cache[checkpoint_path] = model
    return model


class ResultDetr:
    @classmethod
    def result(cls, frame: np.ndarray, checkpoint_path: str, threshold: float) -> list:
        model = _load_model(checkpoint_path)

        orig_h, orig_w = frame.shape[:2]

        img = resize(frame, 640, square=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = infer_transforms(img)

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)

        pred_logits = outputs["pred_logits"][0]  # [num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"][0]    # [num_queries, 4] cxcywh normalized

        probs = F.softmax(pred_logits, dim=-1)   # [num_queries, num_classes]
        scores, labels = probs.max(dim=-1)       # [num_queries]

        keep = (labels != 0) & (scores > threshold)

        results = []
        for score, label, box in zip(scores[keep], labels[keep], pred_boxes[keep]):
            cx, cy, bw, bh = box.cpu().tolist()
            x = (cx - bw / 2) * orig_w
            y = (cy - bh / 2) * orig_h
            w = bw * orig_w
            h = bh * orig_h
            results.append([x, y, w, h, float(score), int(label)])

        return results
