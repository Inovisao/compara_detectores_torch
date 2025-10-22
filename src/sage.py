from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NMS_IOU = 0.5


def detect_sage_dataset(root: str | Path) -> bool:
    """Return True when the tiled dataset was generated with the SAGE tiling mode."""
    root_path = Path(root)
    for split in ("test", "val", "valid", "train"):
        meta_path = root_path / split / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        mode = (
            meta.get("tiling_config", {})
            .get("mode", "")
        )
        if isinstance(mode, str) and mode.lower() == "sage":
            return True
    return False


@dataclass(frozen=True)
class TileInfo:
    split: str
    tile_name: str
    base_key: str
    original_name: str
    offset_x: int
    offset_y: int
    width: int
    height: int


class SageAggregator:
    """Aggregate predictions and annotations from SAGE tiles back to original images."""

    def __init__(self, root: str | Path, fold: str, nms_iou: float = DEFAULT_NMS_IOU) -> None:
        self.root = Path(root)
        self.fold = fold
        self.nms_iou = float(nms_iou)

        self.classes_dict: Dict[int, str] = {}
        self._tile_index: Dict[str, TileInfo] = {}
        self._split_to_images: Dict[str, set[str]] = defaultdict(set)
        self._predictions: Dict[str, List[List[float]]] = defaultdict(list)
        self._active_splits: set[str] = set()

        self._metadata_cache = self._read_metadata()
        self._images_dir, self._base_to_original, classes = self._prepare_original_dataset()
        if classes:
            self.classes_dict = classes

        self._index_tiles()
        if not self.classes_dict:
            # As a last resort, try to read the categories from any tiled annotation json.
            self.classes_dict = self._read_tile_categories()

    # ------------------------------------------------------------------ public API

    def add_tile_prediction(self, tile_name: str, predictions: Iterable[Iterable[float]]) -> None:
        """Accumulate predictions for a given tile."""
        info = self._tile_index.get(tile_name)
        if info is None:
            return
        self._active_splits.add(info.split)
        original = self._base_to_original.get(info.base_key)
        if original is None:
            return

        width = original["width"]
        height = original["height"]
        for pred in predictions or []:
            px, py, pw, ph, plabel, pscore = self._coerce_prediction(pred)
            if pw <= 0 or ph <= 0:
                continue
            gx = px + info.offset_x
            gy = py + info.offset_y
            if gx >= width or gy >= height:
                continue
            clipped_w = min(pw, width - gx)
            clipped_h = min(ph, height - gy)
            if clipped_w <= 0 or clipped_h <= 0:
                continue
            self._predictions[info.original_name].append(
                [gx, gy, clipped_w, clipped_h, plabel, pscore]
            )

    def finalize(
        self,
    ) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], Callable[[str], Optional[str]], Dict[int, str]]:
        """Return the aggregated ground-truth, predictions, resolver and classes."""
        target_split = self._select_split()
        image_names = sorted(self._split_to_images.get(target_split, set()))

        ground_truth: Dict[str, List[List[float]]] = {}
        predictions: Dict[str, List[List[float]]] = {}

        for image_name in image_names:
            base_key = self._image_base_key(image_name)
            original = self._base_to_original.get(base_key)
            if original is None:
                continue
            ground_truth[image_name] = [bbox.copy() for bbox in original["annotations"]]

            merged = self._apply_nms(
                self._predictions.get(image_name, []),
                original["width"],
                original["height"],
            )
            predictions[image_name] = merged

        resolver = self._build_resolver(self._images_dir)
        return ground_truth, predictions, resolver, self.classes_dict

    # ------------------------------------------------------------------ helpers

    def _read_metadata(self) -> Dict[str, dict]:
        cache: Dict[str, dict] = {}
        for split in ("test", "val", "valid", "train"):
            meta_path = self.root / split / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                cache[split] = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
        return cache

    def _prepare_original_dataset(self) -> Tuple[Path, Dict[str, dict], Dict[int, str]]:
        meta = next(iter(self._metadata_cache.values()), {})
        dataset_dir = self._resolve_dataset_dir(meta.get("source_images_dir"))
        annotations_path = self._resolve_annotations_path(meta.get("source_annotations"), dataset_dir)

        classes: Dict[int, str] = {}
        base_to_original: Dict[str, dict] = {}

        if annotations_path.exists():
            with annotations_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            classes = {
                int(cat["id"]): cat["name"]
                for cat in data.get("categories", [])
                if "id" in cat and "name" in cat
            }

            ann_by_image: Dict[int, List[List[float]]] = defaultdict(list)
            for ann in data.get("annotations", []):
                bbox = ann.get("bbox")
                category_id = ann.get("category_id")
                image_id = ann.get("image_id")
                if bbox is None or category_id is None or image_id is None:
                    continue
                ann_by_image[int(image_id)].append(
                    [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), int(category_id)]
                )

            for image in data.get("images", []):
                file_name = image.get("file_name")
                if not file_name:
                    continue
                base_key = self._image_base_key(file_name)
                if not base_key:
                    continue
                width = int(image.get("width", 0))
                height = int(image.get("height", 0))
                base_to_original[base_key] = {
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "annotations": ann_by_image.get(int(image.get("id", -1)), []),
                }

        return dataset_dir, base_to_original, classes

    def _index_tiles(self) -> None:
        for split in ("test", "val", "valid", "train"):
            ann_path = self.root / split / "_annotations.coco.json"
            if not ann_path.exists():
                continue
            try:
                data = json.loads(ann_path.read_text())
            except json.JSONDecodeError:
                continue

            if not self.classes_dict and data.get("categories"):
                self.classes_dict = {
                    int(cat["id"]): cat["name"]
                    for cat in data["categories"]
                    if "id" in cat and "name" in cat
                }

            for image in data.get("images", []):
                tile_name = image.get("file_name")
                if not tile_name or "_tile_" not in tile_name:
                    continue
                base_key, offset_x, offset_y = self._parse_tile_name(tile_name)
                original = self._base_to_original.get(base_key)
                if original is None:
                    continue
                info = TileInfo(
                    split=split,
                    tile_name=tile_name,
                    base_key=base_key,
                    original_name=original["file_name"],
                    offset_x=offset_x,
                    offset_y=offset_y,
                    width=int(image.get("width", 0)),
                    height=int(image.get("height", 0)),
                )
                self._tile_index[tile_name] = info
                self._split_to_images[split].add(original["file_name"])

    def _read_tile_categories(self) -> Dict[int, str]:
        for split in ("test", "val", "valid", "train"):
            ann_path = self.root / split / "_annotations.coco.json"
            if not ann_path.exists():
                continue
            try:
                data = json.loads(ann_path.read_text())
            except json.JSONDecodeError:
                continue
            if data.get("categories"):
                return {
                    int(cat["id"]): cat["name"]
                    for cat in data["categories"]
                    if "id" in cat and "name" in cat
                }
        return {}

    # ------------------------------------------------------------------ static helpers

    @staticmethod
    def _parse_tile_name(name: str) -> Tuple[str, int, int]:
        base, _, coords = name.partition("_tile_")
        coords = coords.rsplit(".", 1)[0]
        parts = coords.split("_")
        try:
            offset_x = int(float(parts[0]))
            offset_y = int(float(parts[1]))
        except (IndexError, ValueError):
            offset_x = 0
            offset_y = 0
        return base, offset_x, offset_y

    @staticmethod
    def _image_base_key(name: str) -> str:
        if not name:
            return ""
        if "_tile_" in name:
            return name.split("_tile_", 1)[0]
        if "_jpg" in name:
            return name.split("_jpg", 1)[0]
        return Path(name).stem

    @staticmethod
    def _resolve_dataset_dir(source_dir: Optional[str]) -> Path:
        default_dir = PROJECT_ROOT / "dataset" / "train"
        if not source_dir:
            return default_dir
        normalized = Path(source_dir.replace("\\", "/"))
        parts = [part for part in normalized.parts if part not in (".", "..")]
        lower_parts = [part.lower() for part in parts]
        if "dataset" in lower_parts:
            idx = len(lower_parts) - 1 - lower_parts[::-1].index("dataset")
            tail = parts[idx + 1 :]
            candidate = PROJECT_ROOT / "dataset"
            for chunk in tail:
                candidate = candidate / chunk
            if candidate.exists():
                return candidate
            if tail:
                fallback = PROJECT_ROOT / "dataset" / tail[-1]
                if fallback.exists():
                    return fallback
        if parts:
            candidate = PROJECT_ROOT / "dataset" / parts[-1]
            if candidate.exists():
                return candidate
        return default_dir

    @staticmethod
    def _resolve_annotations_path(source_ann: Optional[str], dataset_dir: Path) -> Path:
        default_file = dataset_dir / "_annotations.coco.json"
        if not source_ann:
            return default_file
        normalized = Path(source_ann.replace("\\", "/"))
        name = normalized.name
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
        if default_file.exists():
            return default_file
        return candidate

    @staticmethod
    def _coerce_prediction(pred: Iterable[float]) -> Tuple[float, float, float, float, int, float]:
        px, py, pw, ph, plabel, pscore = list(pred)[:6]
        return float(px), float(py), float(pw), float(ph), int(plabel), float(pscore)

    @staticmethod
    def _build_resolver(images_dir: Path) -> Callable[[str], Optional[str]]:
        def resolver(file_name: str) -> Optional[str]:
            path = images_dir / file_name
            return str(path) if path.exists() else None

        return resolver

    def _apply_nms(
        self,
        preds: Iterable[List[float]],
        width: int,
        height: int,
    ) -> List[List[float]]:
        by_class: Dict[int, List[List[float]]] = defaultdict(list)
        for pred in preds:
            if len(pred) < 6:
                continue
            plabel = int(pred[4])
            by_class[plabel].append(pred)

        merged: List[List[float]] = []
        for cls_id, items in by_class.items():
            if not items:
                continue
            scores = [float(item[5]) for item in items]
            boxes = [self._xywh_to_xyxy(item) for item in items]
            keep = self._nms_indices(boxes, scores, self.nms_iou)
            for idx in keep:
                clipped = self._clip_box(items[idx], width, height)
                if clipped is not None:
                    merged.append(clipped)
        return merged

    @staticmethod
    def _xywh_to_xyxy(item: List[float]) -> Tuple[float, float, float, float]:
        x1 = float(item[0])
        y1 = float(item[1])
        x2 = x1 + float(item[2])
        y2 = y1 + float(item[3])
        return x1, y1, x2, y2

    @staticmethod
    def _nms_indices(
        boxes: List[Tuple[float, float, float, float]],
        scores: List[float],
        iou_threshold: float,
    ) -> List[int]:
        order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
        keep: List[int] = []
        while order:
            current = order.pop(0)
            keep.append(current)
            order = [
                idx
                for idx in order
                if SageAggregator._iou(boxes[current], boxes[idx]) < iou_threshold
            ]
        return keep

    @staticmethod
    def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _clip_box(item: List[float], width: int, height: int) -> Optional[List[float]]:
        x, y, w, h, label, score = item
        if w <= 0 or h <= 0:
            return None
        max_w = max(0.0, float(width) - x)
        max_h = max(0.0, float(height) - y)
        w = min(float(w), max_w)
        h = min(float(h), max_h)
        if w <= 0 or h <= 0:
            return None
        return [float(x), float(y), w, h, int(label), float(score)]

    def _select_split(self) -> str:
        if "test" in self._active_splits:
            return "test"
        if self._active_splits:
            return sorted(self._active_splits)[0]
        if "test" in self._split_to_images:
            return "test"
        return next(iter(self._split_to_images.keys()), "test")
