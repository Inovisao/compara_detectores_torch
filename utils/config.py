"""YAML config loading, three-layer merge, and commented config dump."""

from pathlib import Path
from typing import Any, Optional

import yaml

from detectors import DETECTOR_REGISTRY

_DEFAULT_CONFIG = {
    "experiment": "default",
    "dataset": {
        "coco_json": "dataset/all/train/_annotations.coco.json",
        "images_dir": "dataset/all/train",
    },
    "folds": {"n_folds": 5, "val_ratio": 0.2, "seed": 42},
    "output_dir": "results",
    "seed": 42,
    "device": "cuda",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists replace, dicts merge."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> dict:
    """Load and merge configuration: CLI > YAML file > defaults."""
    config = _DEFAULT_CONFIG.copy()

    if yaml_path:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, yaml_config)

    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    return config


def _build_detector_defaults() -> dict[str, Any]:
    """Build per-detector defaults from registered Detector.default_hparams()."""
    detector_defaults = {}
    for name, det_cls in DETECTOR_REGISTRY.items():
        hparams = {}
        for key, info in det_cls.default_hparams().items():
            hparams[key] = info["default"]
        detector_defaults[name] = {
            "architectures": det_cls.architectures(),
            "hparams": hparams,
        }
    return detector_defaults


def dump_commented_config(output_path: str) -> None:
    """Generate a commented YAML config with all registered detectors and their hparams."""
    detector_defaults = _build_detector_defaults()

    lines = [
        "# =============================================================================",
        "# Config-Driven Detector Benchmark — Generated Configuration",
        "# =============================================================================",
        "# All values shown are factory defaults from each detector's implementation.",
        "# Uncomment and edit any parameter to override.",
        "# =============================================================================",
        "",
        "experiment: default",
        "",
        "dataset:",
        f"  coco_json: {_DEFAULT_CONFIG['dataset']['coco_json']}",
        f"  images_dir: {_DEFAULT_CONFIG['dataset']['images_dir']}",
        "",
        "folds:",
        f"  n_folds: {_DEFAULT_CONFIG['folds']['n_folds']}",
        f"  val_ratio: {_DEFAULT_CONFIG['folds']['val_ratio']}",
        f"  seed: {_DEFAULT_CONFIG['folds']['seed']}",
        "",
        f"output_dir: {_DEFAULT_CONFIG['output_dir']}",
        f"seed: {_DEFAULT_CONFIG['seed']}",
        f"device: {_DEFAULT_CONFIG['device']}",
        "",
        "# " + "-" * 68,
        "# DETECTOR HYPERPARAMETERS",
        "# Uncomment any parameter to override the default.",
        "# " + "-" * 68,
        "",
        "detectors:",
    ]

    for name, det_cls in DETECTOR_REGISTRY.items():
        hparams = det_cls.default_hparams()
        arches = det_cls.architectures()
        lines.append(f"  {name}:")
        lines.append(f"    architectures: {arches}")
        lines.append(f"    # hparams: {'-' * 42}")
        for key, info in hparams.items():
            default_val = info["default"]
            help_str = info["help"]
            if isinstance(default_val, str):
                default_str = repr(default_val)
            else:
                default_str = str(default_val)
            pad = " " * max(1, 20 - len(key) - len(default_str))
            lines.append(f"    #   {key}: {default_str}{pad}# {help_str}")
        lines.append("")

    lines.extend([
        "# " + "-" * 68,
        "# SWEEP MODE",
        "# Replace scalar values with lists to create a grid sweep.",
        "# Example:",
        "#   yolov8:",
        "#     architectures: [yolov8n, yolov8s, yolov8m]",
        "#     hparams:",
        "#       lr: [0.0001, 0.001, 0.01]",
        "#       batch_size: [16, 32]",
        "# " + "-" * 68,
    ])

    content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)
