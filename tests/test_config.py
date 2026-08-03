"""Test config loading and merge."""
import tempfile
from utils.config import load_config


def test_load_defaults():
    config = load_config()
    assert config["experiment"] == "default"
    assert config["folds"]["n_folds"] == 5
    assert "coco_json" in config["dataset"]


def test_load_yaml_override():
    yaml_content = "experiment: test_override\nfolds:\n  n_folds: 3\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)
    assert config["experiment"] == "test_override"
    assert config["folds"]["n_folds"] == 3
    assert config["folds"]["val_ratio"] == 0.2


def test_cli_overrides():
    config = load_config(cli_overrides={"experiment": "cli_test", "seed": 999})
    assert config["experiment"] == "cli_test"
    assert config["seed"] == 999


def test_continuation_config_disables_yolo():
    config = load_config("configs/sweeps/faster_rcnn_continue.yaml")
    assert config["resume"] is True
    assert config["detectors"]["yolov8"] is None
    assert config["detectors"]["faster_rcnn"]["hparams"]["batch_size"] <= 4


def test_cuda_allocator_config_avoids_expandable_segments():
    from cli import CUDA_ALLOCATOR_CONFIG

    assert "expandable_segments" not in CUDA_ALLOCATOR_CONFIG
