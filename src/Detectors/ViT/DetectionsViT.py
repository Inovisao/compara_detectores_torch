from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection


class ResultViT:
    _model_cache: Dict[Path, Tuple] = {}

    @classmethod
    def _load_model(cls, model_path: str):
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ViT weights not found at {path}")

        cache_entry = cls._model_cache.get(path)
        if cache_entry:
            return cache_entry

        device = torch.device(os.getenv("VIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model_name = checkpoint["model_name"]
        image_size = checkpoint["image_size"]
        class_names = checkpoint.get("class_names", [])
        category_mapping = checkpoint.get("category_mapping", {})
        # internal_label (0-indexed) → original COCO category_id
        label_to_original = {v: k for k, v in category_mapping.items()} if category_mapping else {}

        id2label = {idx: name for idx, name in enumerate(class_names)}
        label2id = {name: idx for idx, name in id2label.items()}

        config = AutoConfig.from_pretrained(model_name, id2label=id2label, label2id=label2id)
        model = AutoModelForObjectDetection.from_config(config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.to(device)

        image_processor = AutoImageProcessor.from_pretrained(
            model_name, size={"height": image_size, "width": image_size}
        )

        cls._model_cache[path] = (model, image_processor, device, label_to_original)
        return model, image_processor, device, label_to_original

    @classmethod
    def result(cls, frame, model_path: str, threshold: float) -> List[List[float]]:
        model, image_processor, device, label_to_original = cls._load_model(model_path)

        orig_h, orig_w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        encoding = image_processor(images=image_rgb, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        processed = image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        results: List[List[float]] = []
        for box, score, label in zip(processed["boxes"], processed["scores"], processed["labels"]):
            label_val = int(label.item())
            # 0-indexed model label → original COCO category_id
            original_id = label_to_original.get(label_val, label_val)
            x1, y1, x2, y2 = box.tolist()
            results.append([x1, y1, x2 - x1, y2 - y1, original_id, float(score.item())])

        return results
