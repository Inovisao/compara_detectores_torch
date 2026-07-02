from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Detectors.YOLOV8 import config


def test_build_train_kwargs_disables_parallel_workers():
    kwargs = config.build_train_kwargs()

    assert kwargs["workers"] == 0
    assert kwargs["project"] == "YOLOV8"
