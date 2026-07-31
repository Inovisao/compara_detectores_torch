import pytest
import yaml
from pathlib import Path
from automation.config_loader import load_config, ExperimentConfig

def test_load_valid_config(tmp_path):
    config_content = """
experiment_name: "test_exp"
seed: 42
dataset:
  coco_json: "../test/annotations.json"
  images_dir: "../test/images"
  output_dir: "../test/output"
folds:
  n_folds: 5
  val_percentage: 0.3
models:
  YOLOV8:
    architectures: [yolov8s]
    learning_rates: [0.001]
    optimizers: [AdamW]
    batch_sizes: [32]
    weight_decays: [0.0005]
    schedulers: [cosine]
    epochs: [10]
    patience: [5]
execution:
  dry_run: false
  continue_on_error: true
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    
    config = load_config(str(config_file))
    
    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "test_exp"
    assert config.seed == 42
    assert config.folds.n_folds == 5
    assert "YOLOV8" in config.models

def test_load_config_missing_required_field(tmp_path):
    config_content = """
experiment_name: "test_exp"
# missing seed
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    
    with pytest.raises(ValueError, match="Missing required field"):
        load_config(str(config_file))
