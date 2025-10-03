import os
import json
import cv2
from tqdm import tqdm

# === Caminhos ===
DATA_ROOT_DIR = os.path.join("..", "dataset", "all/")
JSON_PATH = os.path.join(DATA_ROOT_DIR, "train/_annotations.coco.json")
IMG_DIR = os.path.join(DATA_ROOT_DIR, "train")
OUTPUT_JSON = os.path.join(IMG_DIR, "_annotations.coco.json")

# === Carregamento ===
with open(JSON_PATH, 'r') as f:
    coco = json.load(f)

# Indexa anotações por imagem_id
anotacoes_por_img = {}
for ann in coco['annotations']:
    anotacoes_por_img.setdefault(ann['image_id'], []).append(ann)

# Inicializações
novas_imagens = []
novas_anotacoes = []
imagens_utilizadas = set()
nova_id = 1

# === Filtra imagens com anotações ===
for img in tqdm(coco['images'], desc="Verificando imagens com anotações"):
    img_id = img['id']
    img_path = os.path.join(IMG_DIR, img['file_name'])
    
    # Verifica se a imagem existe
    if not os.path.exists(img_path):
        continue
    
    # Verifica se há anotações para a imagem
    anns = anotacoes_por_img.get(img_id, [])
    if len(anns) == 0:
        continue

    # Mantém imagem e anotações
    novas_imagens.append(img)
    for ann in anns:
        ann['id'] = nova_id
        ann['image_id'] = img_id
        novas_anotacoes.append(ann)
        nova_id += 1
    imagens_utilizadas.add(img['file_name'])

# === Remove imagens sem anotação ===
for nome_arquivo in os.listdir(IMG_DIR):
    if nome_arquivo.lower().endswith((".jpg", ".jpeg", ".png")) and nome_arquivo not in imagens_utilizadas:
        try:
            os.remove(os.path.join(IMG_DIR, nome_arquivo))
        except Exception as e:
            print(f"⚠️ Erro ao remover {nome_arquivo}: {e}")

# === Salva novo JSON ===
coco_filtrado = {
    "images": novas_imagens,
    "annotations": novas_anotacoes,
    "categories": coco['categories'],
    "info": coco.get('info', {}),
    "licenses": coco.get('licenses', [])
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(coco_filtrado, f, indent=2)

# === Conclusão ===
print(f"✅ Finalizado com sucesso.")
print(f"📸 Imagens mantidas: {len(novas_imagens)}")
print(f"🔎 Anotações mantidas: {len(novas_anotacoes)}")
print(f"🧹 Imagens sem anotação foram removidas de: {IMG_DIR}")
