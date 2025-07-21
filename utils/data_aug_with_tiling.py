import os
import cv2
import json
import shutil
from tqdm import tqdm

DATA_ROOT_DIR = os.path.join("..", "dataset", "all/")

TILE_SIZE = 640
STRIDE = 640  # sem sobreposição
ANNOTATION_PATH = DATA_ROOT_DIR + 'train/_annotations.coco.json'
IMG_DIR = DATA_ROOT_DIR + 'train'
OUTPUT_IMG_DIR = DATA_ROOT_DIR + 'train_tiled'
OUTPUT_JSON = OUTPUT_IMG_DIR + '/_annotations.coco.json'

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

    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_data['id']]

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            x_end = min(x + TILE_SIZE, w)
            y_end = min(y + TILE_SIZE, h)

            if (x_end - x) < TILE_SIZE or (y_end - y) < TILE_SIZE:
                continue  # ignora tiles incompletos (ou mude para preencher)

            tile = image[y:y_end, x:x_end]
            tile_name = f"{os.path.splitext(img_data['file_name'])[0]}_{x}_{y}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, tile_name), tile)

            new_images.append({
                "id": image_id,
                "file_name": tile_name,
                "width": TILE_SIZE,
                "height": TILE_SIZE
            })

            for ann in anns:
                bx, by, bw, bh = ann['bbox']
                box_x2 = bx + bw
                box_y2 = by + bh

                # Verifica se a bbox está completamente dentro da tile
                if bx >= x and by >= y and box_x2 <= x + TILE_SIZE and box_y2 <= y + TILE_SIZE:
                    new_bbox = [bx - x, by - y, bw, bh]

                    new_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": ann['category_id'],
                        "bbox": new_bbox,
                        "area": bw * bh,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            image_id += 1

# Salvar novo COCO JSON
new_coco = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco_data['categories']
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(new_coco, f, indent=4)

print(f"✅ Tiling finalizado: {len(new_images)} imagens criadas e {len(new_annotations)} anotações preservadas.")
