import os
import shutil
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_TILE_ROOT = PROJECT_ROOT / 'dataset' / 'tiles'

# Criar a pasta de destino se não existir
def geredata(fold):
    fold_root = DATASET_TILE_ROOT / fold
    if not fold_root.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_root}")

    destination_folder = fold_root / 'Faster'
    shutil.rmtree(destination_folder, ignore_errors=True)

    for split in ('train', 'val'):
        split_src_dir = fold_root / split
        src_json = split_src_dir / '_annotations.coco.json'
        if not src_json.exists():
            continue
        split_dir = destination_folder / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_json, split_dir / '_annotations.coco.json')

        with open(src_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for imgs in data['images']:
            img_name = imgs['file_name']
            img_path = split_src_dir / img_name
            if not img_path.exists():
                continue
            shutil.copy(img_path, split_dir)
