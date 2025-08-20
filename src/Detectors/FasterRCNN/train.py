import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import os
from tqdm import tqdm
from config import (
    TRAIN_DIR,
    VALID_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    DEVICE,
    LR,
    OUT_DIR,
    PATIENCE,
)

# Define transformations
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# Dataset class
def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

# Load datasets
train_dataset = get_coco_dataset(
    img_dir=TRAIN_DIR,
    ann_file=os.path.join(TRAIN_DIR, '_annotations.coco.json')
)

val_dataset = get_coco_dataset(
    img_dir=VALID_DIR,
    ann_file=os.path.join(VALID_DIR, "_annotations.coco.json")
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

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

# Define optimizer and scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())

    return epoch_loss / len(data_loader)

# Training loop
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

best_loss = float("inf")
patience_counter = 0

for epoch in range(1, NUM_EPOCHS+1):
    loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
    lr_scheduler.step()

    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
        best_model_path = os.path.join(OUT_DIR, 'best.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"Melhor modelo salvo: {best_model_path} com loss {best_loss:.4f}")
    else:
        patience_counter += 1

    if epoch % PATIENCE == 0:
        last_model_path = os.path.join(OUT_DIR, 'last_checkpoint.pth')
        torch.save(model.state_dict(), last_model_path)
        print(f"Modelo salvo: {last_model_path}")

    if patience_counter == PATIENCE:
        print("Parando o treinamento por falta de melhoria.")
        break