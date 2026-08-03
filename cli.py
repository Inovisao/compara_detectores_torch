#!/usr/bin/env python
"""Config-Driven Detector Benchmark CLI."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
CUDA_ALLOCATOR_CONFIG = "max_split_size_mb:128"
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOCATOR_CONFIG
)

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


def _collate_fn(batch):
    return tuple(zip(*batch))


@app.command()
def train(
    detector: str = typer.Option(..., "--detector", "-d", help="Detector family (yolov8, faster_rcnn, detr)"),
    arch: str = typer.Option(..., "--arch", "-a", help="Architecture variant (yolov8s, resnet50)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    fold: int = typer.Option(0, "--fold", "-f", help="Fold index (0=all, 1-based otherwise)"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate override"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Epochs override"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size override"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed override"),
):
    """Train a single detector architecture."""
    from torch.utils.data import DataLoader

    from data.dataset import COCODataset
    from data.transforms import get_train_transforms, get_val_transforms
    from detectors import DETECTOR_REGISTRY
    from utils.config import load_config
    from utils.folds import split_folds

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
        coco["images"], coco["annotations"],
        config["folds"]["n_folds"], config["folds"]["val_ratio"], config["seed"],
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


@app.command(name="eval")
def eval_cmd(
    detector: str = typer.Option(..., "--detector", "-d"),
    weights: Path = typer.Option(..., "--weights", "-w", help="Path to best.pth checkpoint"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    fold: int = typer.Option(1, "--fold", "-f", help="Fold index (1-based)"),
    iou: float = typer.Option(0.2, "--iou", help="IoU threshold for classification matching"),
):
    """Evaluate a trained detector checkpoint on its test fold."""
    from torch.utils.data import DataLoader

    from data.dataset import COCODataset
    from data.transforms import get_val_transforms
    from detectors import DETECTOR_REGISTRY
    from engine.evaluator import append_csv_row, evaluate_detector, save_metrics
    from utils.config import load_config
    from utils.folds import split_folds

    setup_logging("INFO")

    config = load_config(str(config_path) if config_path else None)

    det_cls = DETECTOR_REGISTRY.get(detector)
    if det_cls is None:
        typer.echo(f"Unknown detector '{detector}'. Available: {list(DETECTOR_REGISTRY.keys())}", err=True)
        raise typer.Exit(1)

    with open(config["dataset"]["coco_json"], "r") as f:
        coco = json.load(f)

    folds = split_folds(
        coco["images"], coco["annotations"],
        config["folds"]["n_folds"], config["folds"]["val_ratio"], config["seed"],
    )
    classes = {cat["id"]: cat["name"] for cat in coco["categories"]}

    fi = fold - 1
    if fi < 0 or fi >= len(folds):
        typer.echo(f"Fold {fold} out of range (1-{len(folds)})", err=True)
        raise typer.Exit(1)

    f = folds[fi]
    images_dir = config["dataset"]["images_dir"]
    coco_json = config["dataset"]["coco_json"]
    output_dir = Path(config["output_dir"]) / config.get("experiment", "default")
    fold_dir = output_dir / f"fold_{fold}"

    if not weights.exists():
        arch = weights.parent.name if weights.parent.name != "." else "unknown"
        weights = fold_dir / arch / "best.pth"

    logger.info(f"Fold {fold}/{len(folds)} — weights: {weights}")

    test_ds = COCODataset(coco_json, images_dir, f["test"], get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn)

    det_instance = det_cls()
    det_instance.load(weights)
    metrics = evaluate_detector(det_instance, test_loader, classes, iou)

    fold_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, fold_dir / "metrics.json")
    append_csv_row(output_dir / "summary.csv", {
        "ml": detector, "fold": f"fold_{fold}", **metrics,
    })
    typer.echo(f"Fold {fold}: mAP={metrics['mAP']:.4f} mAP50={metrics['mAP50']:.4f} MAE={metrics['MAE']:.2f}")


@app.command()
def sweep(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to sweep YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print grid without running"),
):
    """Run a hyperparameter sweep."""
    from engine.sweeper import generate_experiment_grid, run_sweep
    from utils.config import load_config

    setup_logging("INFO")
    config = load_config(str(config_path))
    grid = generate_experiment_grid(config)

    if dry_run:
        typer.echo(f"Generated {len(grid)} experiments:")
        for exp in grid:
            typer.echo(f"  {exp['combo_name']} [fold={exp['fold'] + 1}]")
        raise typer.Exit()

    run_sweep(grid, config)


@app.command(name="config")
def config_cmd(
    dump: bool = typer.Option(False, "--dump", help="Write commented default config to stdout"),
):
    """Generate and inspect configuration."""
    from utils.config import dump_commented_config, load_config

    if dump:
        import sys, tempfile
        fd, path = tempfile.mkstemp(suffix=".yaml")
        dump_commented_config(path)
        with open(path, "r") as f:
            sys.stdout.write(f.read())
    else:
        cfg = load_config()
        typer.echo(json.dumps(cfg, indent=2, default=str))


@app.command()
def aggregate(
    results_dir: Path = typer.Argument(..., help="Path to experiment results directory"),
):
    """Compute mean and std across folds."""
    import pandas as pd

    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        typer.echo(f"No summary.csv found in {results_dir}", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(summary_csv)
    metric_cols = [c for c in df.columns if c not in ("detector", "architecture", "fold", "ml")]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]

    group_cols = ["detector", "architecture"] if "detector" in df.columns else ["ml"]

    mean_df = df.groupby(group_cols)[metric_cols].mean().add_suffix("_mean")
    std_df = df.groupby(group_cols)[metric_cols].std().add_suffix("_std")
    result = pd.concat([mean_df, std_df], axis=1).reset_index()

    out_path = results_dir / "summary_aggregated.csv"
    result.to_csv(out_path, index=False)
    typer.echo(f"Aggregated results saved to {out_path}")
    typer.echo(result.to_string())


@app.command()
def analyze(
    results_dir: Path = typer.Option(..., "--results", help="Path to experiment results directory"),
    output_dir: Path = typer.Option(..., "--output", help="Directory for analysis outputs"),
    expected_folds: Optional[int] = typer.Option(
        None, "--expected-folds", help="Expected number of folds"
    ),
):
    """Analyze detector results without changing the raw results."""
    from analysis.pipeline import run_analysis

    paths = run_analysis(results_dir, output_dir, expected_folds)
    for name, path in paths.items():
        typer.echo("%s: %s" % (name, path))


if __name__ == "__main__":
    app()
