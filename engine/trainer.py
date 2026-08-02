"""Generic training loop for torchvision detection models."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
)

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = config.get("optimizer", "SGD").lower()
    lr = config.get("lr", 0.001)
    wd = config.get("weight_decay", 0.0005)
    momentum = config.get("momentum", 0.9)

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_name = config.get("scheduler", "none").lower()
    epochs = config.get("epochs", 100)

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        step_size = config.get("step_size", 10)
        gamma = config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.1
        )
    return None


def _train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc=f"Train epoch {epoch + 1}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        valid = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
        if len(valid) == 0:
            continue
        images = [images[i] for i in valid]
        targets = [targets[i] for i in valid]

        loss_dict = model(images, targets)
        losses = sum(v for v in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{losses.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        valid = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
        if len(valid) == 0:
            continue
        images = [images[i] for i in valid]
        targets = [targets[i] for i in valid]

        loss_dict = model(images, targets)
        losses = sum(v for v in loss_dict.values())
        total_loss += losses.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_torchvision_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    output_dir: Path,
) -> Path:
    """Train a torchvision detection model. Returns path to best checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda"))
    epochs = config.get("epochs", 30)
    patience = config.get("patience", 5)

    model.to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    best_loss = float("inf")
    best_path = output_dir / "best.pth"
    patience_counter = 0
    log_path = output_dir / "logs.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = _validate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}/{epochs} — train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, lr: {current_lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, current_lr])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
                best_path,
            )
        else:
            patience_counter += 1

        last_path = output_dir / "last.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
            last_path,
        )

        if patience > 0 and patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"Training complete. Best model: {best_path}")
    return best_path
