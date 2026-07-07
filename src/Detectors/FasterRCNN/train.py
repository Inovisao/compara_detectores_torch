import os
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
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
    NUM_WORKERS,
    RESIZE_TO,
)


class CocoTransform:
    def __call__(self, image, target):
        width, height = image.size
        image = F.to_tensor(image)
        image = F.resize(image, size=RESIZE_TO)

        scale_x = RESIZE_TO / width
        scale_y = RESIZE_TO / height
        processed_target = []
        for obj in target:
            bbox = obj["bbox"]
            x, y, w, h = bbox
            if w > 0 and h > 0:
                processed_target.append(
                    {
                        **obj,
                        "bbox": [
                            max(0.0, x * scale_x),
                            max(0.0, y * scale_y),
                            max(1.0, w * scale_x),
                            max(1.0, h * scale_y),
                        ],
                    }
                )
        return image, processed_target


def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint_path = os.path.join(OUT_DIR, "best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if "roi_heads.box_predictor" not in k}
        model.load_state_dict(filtered_checkpoint, strict=False)
        print("Pesos parciais carregados com sucesso!")

    return model


def save_checkpoint(model, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", leave=True)

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
        try:
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())
        except RuntimeError as exc:
            error_message = str(exc).lower()
            if "out of memory" in error_message or "cudnn" in error_message or "cuda" in error_message:
                if DEVICE.type == 'cuda':
                    print("[WARN] Falha na CUDA detectada; trocando o treino para CPU.")
                    torch.cuda.empty_cache()
                    model.to(torch.device('cpu'))
                    optimizer = torch.optim.SGD(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=LR,
                        momentum=0.9,
                        weight_decay=0.0005,
                    )
                    return train_one_epoch(model, optimizer, data_loader, torch.device('cpu'), epoch)
                raise
            raise

    if len(data_loader) == 0:
        return 0.0
    return epoch_loss / len(data_loader)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_dataset = get_coco_dataset(
        img_dir=TRAIN_DIR,
        ann_file=os.path.join(TRAIN_DIR, '_annotations.coco.json'),
    )
    val_dataset = get_coco_dataset(
        img_dir=VALID_DIR,
        ann_file=os.path.join(VALID_DIR, '_annotations.coco.json'),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == 'cuda',
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == 'cuda',
    )

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_loss = float("inf")
    patience_counter = 0

    try:
        for epoch in range(NUM_EPOCHS):
            try:
                loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
                lr_scheduler.step()

                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                    best_model_path = save_checkpoint(model, OUT_DIR, 'best.pth')
                    print(f"Melhor modelo salvo: {best_model_path} com loss {best_loss:.4f}")
                else:
                    patience_counter += 1

                last_model_path = save_checkpoint(model, OUT_DIR, 'last_checkpoint.pth')
                print(f"Modelo salvo: {last_model_path}")

                if patience_counter == PATIENCE:
                    print("Parando o treinamento por falta de melhoria.")
                    break
                print(f"[INFO] Época {epoch + 1}/{NUM_EPOCHS} finalizada com sucesso.")
            except RuntimeError as exc:
                error_message = str(exc).lower()
                if "out of memory" in error_message or "cudnn" in error_message or "cuda" in error_message:
                    print(f"[ERRO] Falha durante a época {epoch + 1}: {exc}")
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                        print("[WARN] Memória insuficiente ou incompatibilidade CUDA; o treino continuará em CPU.")
                        model.to(torch.device('cpu'))
                        optimizer = torch.optim.SGD(
                            [p for p in model.parameters() if p.requires_grad],
                            lr=LR,
                            momentum=0.9,
                            weight_decay=0.0005,
                        )
                        continue
                    print("[WARN] Falha CUDA persistente; abortando o treino para evitar reinicializações infinitas.")
                    break
                print(f"[ERRO] Falha durante a época {epoch + 1}: {exc}")
                raise
        if not os.path.exists(os.path.join(OUT_DIR, 'best.pth')):
            save_checkpoint(model, OUT_DIR, 'best.pth')
            print(f"Checkpoint de fallback salvo em {os.path.join(OUT_DIR, 'best.pth')}")
        print("Treinamento FasterRCNN finalizado com sucesso!")
    except FileNotFoundError as exc:
        print(f"[ERRO] Arquivo não encontrado: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERRO FATAL] Falha no treinamento FasterRCNN: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()