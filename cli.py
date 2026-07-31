#!/usr/bin/env python
"""Config-Driven Detector Benchmark CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from torch.utils.data import DataLoader

from utils.logging import setup_logging
from utils.config import load_config
from utils.folds import split_folds
from data.dataset import COCODataset
from data.transforms import get_train_transforms, get_val_transforms
from detectors import DETECTOR_REGISTRY

app = typer.Typer()
logger = logging.getLogger(__name__)


def _collate_fn(batch):
    return tuple(zip(*batch))


@app.command()
def train(
    detector: str = typer.Option(..., "--detector", "-d", help="Detector family name (e.g. yolov8, faster_rcnn)"),
    arch: str = typer.Option(..., "--arch", "-a", help="Architecture variant (e.g. yolov8s, resnet50)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    fold: int = typer.Option(0, "--fold", "-f", help="Fold index (0-based, 0 = run all folds)"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate override"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Epochs override"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size override"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed override"),
):
    """Train a single detector architecture."""
    setup_logging("INFO")

    cli_overrides = {"seed": seed} if seed is not None else {}
    if lr is not None or epochs is not None or batch_size is not None:
        cli_overrides["detectors"] = {detector: {"hparams": {}}}
        if lr is not None:
            cli_overrides["detectors"][detector]["hparams"]["lr"] = lr
        if epochs is not None:
            cli_overrides["detectors"][detector]["hparams"]["epochs"] = epochs
        if batch_size is not None:
            cli_overrides["detectors"][detector]["hparams"]["batch_size"] = batch_size

    config = load_config(str(config_path) if config_path else None, cli_overrides)

    det_cls = DETECTOR_REGISTRY.get(detector)
    if det_cls is None:
        typer.echo(f"Unknown detector '{detector}'. Available: {list(DETECTOR_REGISTRY.keys())}", err=True)
        raise typer.Exit(1)
    if arch not in det_cls.architectures():
        typer.echo(f"Unknown architecture '{arch}' for {detector}. Available: {det_cls.architectures()}", err=True)
        raise typer.Exit(1)

    det_hparams = det_cls.default_hparams()
    run_config = {key: info["default"] for key, info in det_hparams.items()}

    if "detectors" in config and detector in config["detectors"]:
        det_config = config["detectors"][detector]
        if "hparams" in det_config:
            run_config.update(det_config["hparams"])

    with open(config["dataset"]["coco_json"], "r") as f:
        coco = json.load(f)

    folds = split_folds(
        coco["images"],
        coco["annotations"],
        config["folds"]["n_folds"],
        config["folds"]["val_ratio"],
        config["seed"],
    )

    run_config["num_classes"] = len(coco["categories"])
    run_config["architecture"] = arch
    run_config["device"] = config.get("device", "cuda")

    images_dir = config["dataset"]["images_dir"]
    coco_json = config["dataset"]["coco_json"]
    img_size = run_config.get("imgsz", 640)

    fold_indices = range(len(folds)) if fold == 0 else [fold - 1]

    for fi in fold_indices:
        f = folds[fi]
        output_dir = (
            Path(config["output_dir"])
            / config.get("experiment", "default")
            / f"fold_{fi + 1}"
            / arch
        )
        logger.info(f"Fold {fi + 1}/{len(folds)} — output: {output_dir}")

        train_ds = COCODataset(coco_json, images_dir, f["train"], get_train_transforms(img_size))
        val_ds = COCODataset(coco_json, images_dir, f["val"], get_val_transforms(img_size))
        train_loader = DataLoader(train_ds, batch_size=run_config["batch_size"], shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=run_config["batch_size"], shuffle=False, collate_fn=_collate_fn)

        det_instance = det_cls()
        best_path = det_instance.train(train_loader, val_loader, run_config, output_dir)
        logger.info(f"Fold {fi + 1} complete — best model: {best_path}")

    typer.echo("Training complete.")


if __name__ == "__main__":
    app()
