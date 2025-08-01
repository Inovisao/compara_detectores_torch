import os
import cv2
import json
from tqdm import tqdm

DATA_ROOT_DIR = os.path.join("..", "dataset", "all/")

ANNOTATION_PATH = os.path.join(DATA_ROOT_DIR, "train/_annotations.coco.json")
IMG_DIR = os.path.join(DATA_ROOT_DIR, "train")
OUTPUT_IMG_DIR = os.path.join(DATA_ROOT_DIR, "train_tiled")
OUTPUT_JSON = os.path.join(OUTPUT_IMG_DIR, "_annotations.coco.json")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

with open(ANNOTATION_PATH, 'r') as f:
    coco_data = json.load(f)

new_images = []
new_annotations = []
annotation_id = 1
image_id = 1

for img_data in tqdm(coco_data['images'], desc='Processando imagens'):
    img_path = os.path.join(IMG_DIR, img_data['file_name'])
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Erro ao ler imagem: {img_path}")
        continue

    h, w, _ = image.shape
    TILE_H = h // 2
    TILE_W = w // 2

    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_data['id']]

    # 4 quadrantes: (0,0), (0,1), (1,0), (1,1)
    for row in range(2):
        for col in range(2):
            x = col * TILE_W
            y = row * TILE_H
            tile = image[y:y+TILE_H, x:x+TILE_W]
            tile_name = f"{os.path.splitext(img_data['file_name'])[0]}_{x}_{y}.jpg"

            anns_tile = []
            for ann in anns:
                bx, by, bw, bh = ann['bbox']
                box_x2 = bx + bw
                box_y2 = by + bh

                # Verifica se bbox está totalmente dentro do tile
                if bx >= x and by >= y and box_x2 <= x + TILE_W and box_y2 <= y + TILE_H:
                    new_bbox = [bx - x, by - y, bw, bh]
                    anns_tile.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": ann['category_id'],
                        "bbox": new_bbox,
                        "area": bw * bh,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            if anns_tile:
                # Só salva se houver pelo menos uma anotação
                cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, tile_name), tile)

                new_images.append({
                    "id": image_id,
                    "file_name": tile_name,
                    "width": TILE_W,
                    "height": TILE_H
                })

                new_annotations.extend(anns_tile)
                image_id += 1

# Salvar novo JSON
new_coco = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco_data['categories']
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(new_coco, f, indent=4)

print(f"✅ Slicing 2x2 finalizado: {len(new_images)} tiles salvos com {len(new_annotations)} anotações.")
