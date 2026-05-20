from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Detection = Tuple[float, float, float, float, int, float]  # x1,y1,x2,y2,class_id,score


class SSDLitePredictor:
    """
    Carrega modelo ONNX + âncoras e expõe predict(image_bgr).

    Parâmetros
    ----------
    onnx_path      : caminho para o arquivo .onnx gerado por export_onnx.py
    anchors_path   : caminho para o .anchors.npy (padrão: mesmo nome do .onnx)
    score_thresh   : confiança mínima para manter uma detecção
    nms_thresh     : limiar IoU para o NMS
    max_detections : número máximo de caixas retornadas
    """

    _IMAGE_SIZE = 320
    # BoxCoder padrão do torchvision: pesos [wx, wy, ww, wh]
    _BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

    def __init__(
        self,
        onnx_path: str | Path,
        anchors_path: str | Path | None = None,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if anchors_path is None:
            anchors_path = onnx_path.parent / (onnx_path.stem + ".anchors.npy")

        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._anchors: np.ndarray = np.load(str(anchors_path))  # (N, 4) cx cy w h
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections

    # ──────────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        """
        Recebe frame BGR (como retornado por cv2.imread) e devolve uma lista
        de detecções [(x1, y1, x2, y2, class_id, score)] em coordenadas de
        pixel da imagem original.
        """
        orig_h, orig_w = image_bgr.shape[:2]
        blob = self._preprocess(image_bgr)

        bbox_regression, cls_logits = self._session.run(None, {"image": blob})
        # bbox_regression : (1, N_anchors, 4)
        # cls_logits      : (1, N_anchors, num_classes)

        return self._postprocess(
            bbox_regression[0], cls_logits[0], orig_w, orig_h
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Pré e pós-processamento
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        s = self._IMAGE_SIZE
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (s, s))
        blob = img.astype(np.float32) / 255.0          # [0, 1]
        blob = blob.transpose(2, 0, 1)[np.newaxis]     # (1, 3, H, W)
        # A normalização ImageNet está embutida no grafo ONNX — não aplicar aqui.
        return blob

    def _postprocess(
        self,
        bbox_regression: np.ndarray,   # (N_anchors, 4)
        cls_logits: np.ndarray,         # (N_anchors, num_classes)
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        boxes_xyxy = self._decode_boxes(bbox_regression, self._anchors)
        np.clip(boxes_xyxy, 0.0, 1.0, out=boxes_xyxy)

        # Softmax numericamente estável
        shifted = cls_logits - cls_logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)  # (N, num_classes)

        detections: List[Detection] = []
        num_classes = cls_logits.shape[1]

        for cls_id in range(1, num_classes):  # 0 = background
            scores = probs[:, cls_id]
            mask = scores >= self.score_thresh
            if not mask.any():
                continue

            cls_boxes = boxes_xyxy[mask]
            cls_scores = scores[mask]
            keep = _nms(cls_boxes, cls_scores, self.nms_thresh)

            for idx in keep:
                x1, y1, x2, y2 = cls_boxes[idx]
                detections.append((
                    float(x1 * orig_w),
                    float(y1 * orig_h),
                    float(x2 * orig_w),
                    float(y2 * orig_h),
                    cls_id,
                    float(cls_scores[idx]),
                ))

        detections.sort(key=lambda d: d[5], reverse=True)
        return detections[: self.max_detections]

    def _decode_boxes(
        self, pred: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """Decodifica regressões SSD → caixas (x1, y1, x2, y2) normalizadas."""
        wx, wy, ww, wh = self._BOX_WEIGHTS
        cx = pred[:, 0] / wx * anchors[:, 2] + anchors[:, 0]
        cy = pred[:, 1] / wy * anchors[:, 3] + anchors[:, 1]
        w  = np.exp(np.clip(pred[:, 2] / ww, -4.0, 4.0)) * anchors[:, 2]
        h  = np.exp(np.clip(pred[:, 3] / wh, -4.0, 4.0)) * anchors[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# NMS em numpy puro (sem dependência extra)
# ──────────────────────────────────────────────────────────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    order = np.argsort(scores)[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: List[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-8)
        order = rest[iou <= iou_thresh]
    return keep
