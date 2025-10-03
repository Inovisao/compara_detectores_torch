import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent / "tph-yolov5"
_IMPORT_ERROR: Optional[Exception] = None

if REPO_DIR.exists():
    repo_path = str(REPO_DIR)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)

try:
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - captured for diagnostics
    attempt_load = None  # type: ignore
    _IMPORT_ERROR = exc


MOSTRAIMAGE = False


def xyxy_to_xywh(boxes: List[List[float]]) -> List[List[float]]:
    coco_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, conf, cls])
    return coco_boxes


class ResultYOLOV5TPH:
    _model_cache = {}

    @classmethod
    def _ensure_repo_available(cls) -> None:
        if attempt_load is None:
            details = f" ({_IMPORT_ERROR})" if _IMPORT_ERROR else ""
            raise ImportError(
                "Cannot import YOLOv5 TPH modules. Clone the tph-yolov5 repository into "
                "src/Detectors/YOLOV5_TPH/tph-yolov5 before running inference." + details
            )

    @classmethod
    def _load_model(cls, model_path: str):
        cls._ensure_repo_available()
        cached = cls._model_cache.get(model_path)
        if cached:
            return cached
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(model_path, map_location=device)  # type: ignore[arg-type]
        model.eval()
        cls._model_cache[model_path] = (model, device)
        return model, device

    @classmethod
    def result(cls, frame, model_path: str, threshold: float):
        model, device = cls._load_model(model_path)
        img0 = frame.copy()
        stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
        img = letterbox(img0, 640, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, threshold, 0.45, None, False)[0]

        results = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred.cpu().numpy():
                x1, y1, x2, y2 = xyxy
                results.append([int(x1), int(y1), int(x2), int(y2), int(cls + 1), float(conf)])
        return xyxy_to_xywh(results)
