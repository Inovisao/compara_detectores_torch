from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.transforms.functional import to_tensor


class ResultRetinaNet:
    _model_cache: Dict[Path, Tuple[torch.nn.Module, torch.device, List[str], Dict[int, int]]] = {}

    @classmethod
    def _load_model(cls, model_path: str):
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"RetinaNet weights not found at {path}")

        cache_entry = cls._model_cache.get(path)
        if cache_entry:
            return cache_entry

        device = torch.device(os.getenv("RETINANET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        num_classes = int(checkpoint.get("num_classes", 1))
        class_names = checkpoint.get("class_names", [str(i) for i in range(1, num_classes + 1)])
        category_mapping = checkpoint.get("category_mapping", {})
        # internal_label (1-indexed) → original COCO category_id
        label_to_original = {v: k for k, v in category_mapping.items()} if category_mapping else {}

        model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.to(device)

        cls._model_cache[path] = (model, device, class_names, label_to_original)
        return model, device, class_names, label_to_original

    @classmethod
    def result(cls, frame, model_path: str, threshold: float):
        model, device, _, label_to_original = cls._load_model(model_path)
        image_tensor = to_tensor(frame).to(device)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()

        keep = scores >= threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        results: List[List[float]] = []
        for box, score, label in zip(boxes, scores, labels):
            label_val = int(label.item())
            if label_val <= 0:
                continue  # skip background
            original_id = label_to_original.get(label_val, label_val)
            x1, y1, x2, y2 = box.tolist()
            results.append([x1, y1, x2 - x1, y2 - y1, original_id, float(score.item())])

        return results
