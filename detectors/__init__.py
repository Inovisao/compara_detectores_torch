from detectors.base import Detector
from detectors.faster_rcnn import FasterRCNNDetector
from detectors.yolov8 import YOLOV8Detector
from detectors.detr import DETRDetector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "faster_rcnn": FasterRCNNDetector,
    "yolov8": YOLOV8Detector,
    "detr": DETRDetector,
}
