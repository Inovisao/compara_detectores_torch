from detectors.base import Detector
from detectors.faster_rcnn import FasterRCNNDetector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "faster_rcnn": FasterRCNNDetector,
}
