"""
Pré-processa o dataset original para 320×320, mantendo a proporção via
LongestMaxSize + PadIfNeeded (padding preto centralizado).

Uso:
    python utils/preprocess_dataset.py
    python utils/preprocess_dataset.py --input dataset/all --output dataset/all_320 --size 320

Saída:
    dataset/all_320/train/          — imagens redimensionadas
    dataset/all_320/train/_annotations.coco.json — JSON com coordenadas atualizadas

Depois rode geraDobras.py apontando para o novo JSON:
    python utils/geraDobras.py -annotations dataset/all_320/train/_annotations.coco.json \
                               -json dataset/all_320/filesJSON/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import albumentations as A
import cv2


def _build_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=0,
                value=0,
                position="center",
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.1,
        ),
    )


def _process_image(
    image_path: Path,
    annotations: list[dict],
    transform: A.Compose,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    """
    Processa uma imagem e retorna (image_info atualizado, anotações atualizadas).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_h, img_w = image.shape[:2]

    boxes_xyxy, labels, ann_ids, areas, iscrowds = [], [], [], [], []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(img_w), x + w)
        y2 = min(float(img_h), y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes_xyxy.append([x1, y1, x2, y2])
        labels.append(ann["category_id"])
        ann_ids.append(ann["id"])
        areas.append(ann.get("area", w * h))
        iscrowds.append(ann.get("iscrowd", 0))

    result = transform(
        image=image,
        bboxes=boxes_xyxy,
        labels=labels,
    )

    out_image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), out_image)

    out_h, out_w = out_image.shape[:2]
    image_info = {
        "id": None,          # preenchido pelo chamador
        "file_name": image_path.name,
        "width": out_w,
        "height": out_h,
        "license": 1,
        "date_captured": "",
    }

    # reconstrói anotações com coordenadas transformadas
    # result["bboxes"] pode ter menos boxes (min_visibility filtrou)
    # labels e ann_ids correspondentes também foram filtrados pelo albumentations
    out_annotations = []
    for (x1, y1, x2, y2), cat_id, orig_ann_id, iscrowd in zip(
        result["bboxes"],
        result["labels"],
        ann_ids[: len(result["bboxes"])],
        iscrowds[: len(result["bboxes"])],
    ):
        bw = x2 - x1
        bh = y2 - y1
        out_annotations.append({
            "id": orig_ann_id,
            "image_id": None,   # preenchido pelo chamador
            "category_id": cat_id,
            "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
            "area": round(bw * bh, 2),
            "iscrowd": iscrowd,
        })

    return image_info, out_annotations


def preprocess(input_dir: Path, output_dir: Path, size: int) -> None:
    images_src = input_dir / "train"
    ann_src    = images_src / "_annotations.coco.json"

    if not ann_src.exists():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {ann_src}")

    images_dst = output_dir / "train"
    images_dst.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] Lendo anotações: {ann_src}")
    with open(ann_src, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # índice: image_id → lista de anotações
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    transform = _build_transform(size)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    skipped = 0

    total = len(coco["images"])
    for i, img_info in enumerate(coco["images"], start=1):
        image_path = images_src / img_info["file_name"]
        if not image_path.exists():
            print(f"[WARN] imagem não encontrada, pulando: {image_path}")
            skipped += 1
            continue

        image_id = int(img_info["id"])
        annotations = anns_by_image.get(image_id, [])

        try:
            new_img_info, new_anns = _process_image(
                image_path, annotations, transform, images_dst
            )
        except Exception as exc:
            print(f"[WARN] erro ao processar {image_path.name}: {exc}")
            skipped += 1
            continue

        new_img_info["id"] = image_id
        for ann in new_anns:
            ann["image_id"] = image_id

        out_images.append(new_img_info)
        out_annotations.extend(new_anns)

        if i % 200 == 0 or i == total:
            print(f"[preprocess] {i}/{total} processadas...", flush=True)

    out_coco = {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  coco["categories"],
        "images":      out_images,
        "annotations": out_annotations,
    }

    out_ann_path = images_dst / "_annotations.coco.json"
    with open(out_ann_path, "w", encoding="utf-8") as f:
        json.dump(out_coco, f, indent=2, ensure_ascii=False)

    print(f"\n[preprocess] Concluído.")
    print(f"  Imagens processadas : {len(out_images)}")
    print(f"  Anotações mantidas  : {len(out_annotations)}")
    print(f"  Puladas/com erro    : {skipped}")
    print(f"  JSON salvo em       : {out_ann_path}")
    print(f"\nPróximo passo:")
    print(f"  python utils/geraDobras.py \\")
    print(f"      -annotations {out_ann_path} \\")
    print(f"      -json {output_dir}/filesJSON/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processa dataset para tamanho fixo")
    parser.add_argument(
        "--input", default="dataset/all", type=Path,
        help="Pasta raiz do dataset original (contém train/)",
    )
    parser.add_argument(
        "--output", default="dataset/all_320", type=Path,
        help="Pasta raiz de saída",
    )
    parser.add_argument(
        "--size", default=320, type=int,
        help="Tamanho alvo (lado máximo), padrão 320",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    preprocess(root / args.input, root / args.output, args.size)
