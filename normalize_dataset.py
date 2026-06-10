"""
Renomeia arquivos de imagem com extensão maiúscula (.JPG, .PNG, etc.) para
minúscula (.jpg, .png) e atualiza todos os JSONs de anotações COCO.

Uso:
    python normalize_dataset.py [--dry-run]
"""
import argparse
import json
import os
from pathlib import Path

DATASET_ROOT = Path(__file__).parent / "dataset" / "all"
IMAGE_DIR = DATASET_ROOT / "train"
JSON_FILES = [
    DATASET_ROOT / "train" / "_annotations.coco.json",
    *sorted((DATASET_ROOT / "filesJSON").glob("*.json")),
]


def normalize_images(dry_run: bool) -> dict:
    """Renomeia arquivos com extensão maiúscula. Retorna mapa old_name → new_name."""
    renames = {}
    for path in IMAGE_DIR.iterdir():
        if path.suffix != path.suffix.lower():
            new_path = path.with_suffix(path.suffix.lower())
            if new_path.exists():
                print(f"  [SKIP] conflito: {path.name} → {new_path.name} já existe")
                continue
            renames[path.name] = new_path.name
            if not dry_run:
                path.rename(new_path)
            print(f"  {'[dry]' if dry_run else '[ok]'} {path.name} → {new_path.name}")
    return renames


def update_json(json_path: Path, renames: dict, dry_run: bool) -> int:
    """Atualiza file_name nas imagens do JSON. Retorna número de substituições."""
    with open(json_path) as f:
        data = json.load(f)

    count = 0
    for img in data.get("images", []):
        old = img["file_name"]
        if old in renames:
            img["file_name"] = renames[old]
            count += 1

    if count and not dry_run:
        with open(json_path, "w") as f:
            json.dump(data, f)

    print(f"  {'[dry]' if dry_run else '[ok]'} {json_path.name}: {count} substituições")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria feito")
    args = parser.parse_args()

    print(f"\n=== Normalizando imagens em {IMAGE_DIR} ===")
    renames = normalize_images(args.dry_run)
    print(f"Total de arquivos renomeados: {len(renames)}\n")

    if not renames:
        print("Nenhuma renomeação necessária.")
        return

    print("=== Atualizando JSONs ===")
    total = 0
    for json_path in JSON_FILES:
        if json_path.exists():
            total += update_json(json_path, renames, args.dry_run)
        else:
            print(f"  [SKIP] {json_path.name} não encontrado")

    print(f"\nTotal de entradas atualizadas nos JSONs: {total}")
    if args.dry_run:
        print("\n[dry-run] Nenhuma alteração foi feita. Rode sem --dry-run para aplicar.")


if __name__ == "__main__":
    main()
