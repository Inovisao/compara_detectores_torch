"""Sweep engine: grid generation and train+eval orchestration."""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

from torch.utils.data import DataLoader

from data.dataset import COCODataset
from data.transforms import get_train_transforms, get_val_transforms
from detectors import DETECTOR_REGISTRY
from engine.evaluator import evaluate_detector, save_metrics, append_csv_row
from utils.folds import split_folds

logger = logging.getLogger(__name__)


def _is_list(v):
    return isinstance(v, list)


def _collate_fn(batch):
    return tuple(zip(*batch))


def generate_experiment_grid(config: dict) -> list[dict]:
    """Generate all experiment combinations from sweep config.

    Scalar hparams are fixed; list hparams are swept (cartesian product).
    """
    experiments = []
    experiment_name = config.get("experiment", "sweep")
    base_seed = config.get("seed", 42)
    n_folds = config["folds"]["n_folds"]
    detectors = config.get("detectors", {})

    with open(config["dataset"]["coco_json"], "r") as f:
        coco = json.load(f)
    num_classes = len(coco["categories"])

    folds = split_folds(
        coco["images"], coco["annotations"],
        n_folds, config["folds"]["val_ratio"], base_seed,
    )

    for det_name, det_config in detectors.items():
        if det_config is None:
            continue
        det_cls = DETECTOR_REGISTRY.get(det_name)
        if det_cls is None:
            logger.warning(f"Unknown detector: {det_name}, skipping")
            continue

        hparams = det_config.get("hparams", {})
        sweep_keys = [k for k, v in hparams.items() if _is_list(v)]
        fixed_keys = [k for k, v in hparams.items() if not _is_list(v)]

        hparams_full = {key: info["default"] for key, info in det_cls.default_hparams().items()}
        for k in fixed_keys:
            hparams_full[k] = hparams[k]

        architectures = det_config.get("architectures", det_cls.architectures())

        sweep_values = []
        for key in sweep_keys:
            sweep_values.append([(key, v) for v in hparams[key]])
        if not sweep_values:
            sweep_values = []

        for arch in architectures:
            for sweep_combo in itertools.product(*sweep_values):
                hps = dict(hparams_full)
                for key, val in sweep_combo:
                    hps[key] = val

                for fi in range(n_folds):
                    combo_parts = [det_name, arch]
                    combo_parts.extend([f"{k}={v}" for k, v in sweep_combo])
                    combo_name = "_".join(combo_parts)

                    experiments.append({
                        "detector": det_name,
                        "architecture": arch,
                        "hparams": hps,
                        "fold": fi,
                        "num_classes": num_classes,
                        "experiment": experiment_name,
                        "combo_name": combo_name,
                    })

    return experiments


def run_sweep(experiments: list[dict], config: dict) -> None:
    """Run all experiments sequentially."""
    images_dir = config["dataset"]["images_dir"]
    coco_json = config["dataset"]["coco_json"]

    with open(coco_json, "r") as f:
        coco = json.load(f)
    classes = {cat["id"]: cat["name"] for cat in coco["categories"]}
    n_folds = config["folds"]["n_folds"]
    base_seed = config.get("seed", 42)

    folds = split_folds(
        coco["images"], coco["annotations"],
        n_folds, config["folds"]["val_ratio"], base_seed,
    )

    output_dir = Path(config["output_dir"]) / config.get("experiment", "sweep")

    for exp in experiments:
        det_name = exp["detector"]
        arch = exp["architecture"]
        fi = exp["fold"]
        hps = exp["hparams"]
        hps["architecture"] = arch
        hps["num_classes"] = exp["num_classes"]
        hps["device"] = config.get("device", "cuda")
        img_size = hps.get("imgsz", 640)

        fold_dir = output_dir / f"fold_{fi + 1}" / arch
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{det_name}/{arch}] Fold {fi + 1}/{n_folds}")

        det_instance = DETECTOR_REGISTRY[det_name]()

        f = folds[fi]
        train_ds = COCODataset(coco_json, images_dir, f["train"], get_train_transforms(img_size))
        val_ds = COCODataset(coco_json, images_dir, f["val"], get_val_transforms(img_size))
        train_loader = DataLoader(train_ds, batch_size=hps.get("batch_size", 8), shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=hps.get("batch_size", 8), shuffle=False, collate_fn=_collate_fn)

        best_path = det_instance.train(train_loader, val_loader, hps, fold_dir)

        test_ds = COCODataset(coco_json, images_dir, f["test"], get_val_transforms(img_size))
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn)

        det_instance.load(best_path)
        metrics = evaluate_detector(det_instance, test_loader, classes)

        save_metrics(metrics, fold_dir / "metrics.json")
        append_csv_row(output_dir / "summary.csv", {
            "detector": det_name,
            "architecture": arch,
            "fold": f"fold_{fi + 1}",
            **{k: v for k, v in hps.items() if k not in ("architecture", "num_classes", "device")},
            **metrics,
        })

    logger.info(f"Sweep complete. Results: {output_dir}")
