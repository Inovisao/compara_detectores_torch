from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from ultralytics import YOLO


MOSTRAIMAGE = False


def _xyxy_to_xywh(boxes: np.ndarray) -> List[List[float]]:
    coco_boxes: List[List[float]] = []
    for x_min, y_min, x_max, y_max, conf, cls in boxes:
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([float(x_min), float(y_min), float(w), float(h), int(cls), float(conf)])
    return coco_boxes


class ResultYOLOV11:
    _model_cache: dict[str, Tuple[YOLO, str]] = {}

    @classmethod
    def _load_model(cls, model_path: str) -> Tuple[YOLO, str]:
        path = Path(model_path).expanduser()
        if path.exists():
            source = str(path.resolve())
            cache_key = f"{source}:{path.stat().st_mtime_ns}"
        else:
            source = model_path  # allow Ultralytics to download by alias
            cache_key = source

        cached = cls._model_cache.get(cache_key)
        if cached:
            return cached

        model = YOLO(source)
        model.fuse()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._model_cache[cache_key] = (model, device)
        return model, device

    @classmethod
    def result(cls, frame, model_path: str, threshold: float):
        model, device = cls._load_model(model_path)
        results = model.predict(frame, conf=threshold, device=device, verbose=False)
        if not results:
            return []

        det = results[0]
        if det.boxes is None or det.boxes.xyxy.numel() == 0:
            return []

        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int) + 1  # 1-based classes
        stacked = np.hstack([boxes, scores[:, np.newaxis], classes[:, np.newaxis]])
        return _xyxy_to_xywh(stacked)
