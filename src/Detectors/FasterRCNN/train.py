import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import os
import sys
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from config import (
    TRAIN_DIR,
    TRAIN_ANN_PATH,
    VALID_DIR,
    VAL_ANN_PATH,
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    RESIZE_TO,
    DEVICE,
    LR,
    MOMENTUM,
    WEIGHT_DECAY,
    OUT_DIR,
    PATIENCE,
)

_AUG_PIPELINE = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=round(0.015 * 180),
            sat_shift_limit=round(0.7 * 100),
            val_shift_limit=round(0.4 * 100),
            p=1.0,
        ),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.3),
)


class CocoTransform:
    def __init__(self, augment: bool = False):
        self.augment = augment

    def __call__(self, image, target):
        # Resize mantendo aspect ratio
        img_np = np.array(image)  # PIL → RGB numpy
        h, w = img_np.shape[:2]
        scale = RESIZE_TO / max(h, w)
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_np = cv2.resize(img_np, (new_w, new_h))
            if target:
                for obj in target:
                    x, y, bw, bh = obj["bbox"]
                    obj["bbox"] = [x * scale, y * scale, bw * scale, bh * scale]

        if self.augment and target:
            bboxes = [obj["bbox"] for obj in target]
            labels = [obj["category_id"] for obj in target]
            try:
                result = _AUG_PIPELINE(image=img_np, bboxes=bboxes, class_labels=labels)
                img_np = result["image"]
                # Reconstrói target a partir do resultado — albumentations pode remover
                # bboxes por min_visibility, quebrando o mapeamento 1:1 por índice.
                target = [
                    {"bbox": list(box), "category_id": int(cat)}
                    for box, cat in zip(result["bboxes"], result["class_labels"])
                ]
            except Exception:
                pass

        image = F.to_tensor(F.to_pil_image(img_np))
        return image, target


# Dataset class
def get_coco_dataset(img_dir, ann_file, augment=False):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform(augment=augment)
    )

# Validate dataset configuration
if not all([TRAIN_DIR, TRAIN_ANN_PATH, VALID_DIR, VAL_ANN_PATH]):
    raise RuntimeError(
        "FasterRCNN dataset paths are not configured. "
        "Ensure runFaster sets FASTER_* environment variables before launching training."
    )

# Load datasets
train_dataset = get_coco_dataset(
    img_dir=TRAIN_DIR,
    ann_file=TRAIN_ANN_PATH,
    augment=True,
)

val_dataset = get_coco_dataset(
    img_dir=VALID_DIR,
    ann_file=VAL_ANN_PATH,
    augment=False,
)

# DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    collate_fn=lambda x: tuple(zip(*x)),
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    collate_fn=lambda x: tuple(zip(*x)),
)

# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint_path = 'bestFaster.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if "roi_heads.box_predictor" not in k}
        model.load_state_dict(filtered_checkpoint, strict=False)
        print("Pesos parciais carregados com sucesso!")
    
    return model

# Initialize the model
model = get_model(NUM_CLASSES)
model.to(DEVICE)
if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
    except Exception:
        pass

# Define optimizer and scheduler
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

_scaler = torch.amp.GradScaler("cuda", enabled=str(DEVICE).startswith("cuda"))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=True)

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                bbox = obj["bbox"]
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(obj["category_id"])
            if boxes:
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i])

        if not processed_targets:
            continue

        images = valid_images
        with torch.amp.autocast("cuda", enabled=_scaler.is_enabled()):
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        _scaler.scale(losses).backward()
        _scaler.step(optimizer)
        _scaler.update()

        epoch_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())

    return epoch_loss / len(data_loader)

# Training loop
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[FasterRCNN] OUT_DIR={os.path.abspath(OUT_DIR)}", flush=True)

best_loss = float("inf")
patience_counter = 0

try:
    for epoch in range(NUM_EPOCHS):
        try:
            loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
            lr_scheduler.step()

            os.makedirs(OUT_DIR, exist_ok=True)
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                best_model_path = os.path.join(OUT_DIR, 'best.pth')
                state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                torch.save(state_dict, best_model_path)
                print(f"Melhor modelo salvo: {best_model_path} com loss {best_loss:.4f}")
            else:
                patience_counter += 1

            last_model_path = os.path.join(OUT_DIR, 'last_checkpoint.pth')
            state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save(state_dict, last_model_path)
            print(f"Modelo salvo: {last_model_path}")

            if patience_counter == PATIENCE:
                print("Parando o treinamento por falta de melhoria.")
                break
            print(f"[INFO] Época {epoch+1}/{NUM_EPOCHS} finalizada com sucesso.")
        except Exception as e:
            print(f"[ERRO] Falha durante a época {epoch+1}: {e}")
            continue
    print("Treinamento FasterRCNN finalizado com sucesso!")
except FileNotFoundError as e:
    print(f"[ERRO] Arquivo não encontrado: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERRO FATAL] Falha no treinamento FasterRCNN: {e}")
    sys.exit(1)
