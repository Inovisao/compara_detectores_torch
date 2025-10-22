#!/usr/bin/env python3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dataset_path = PROJECT_ROOT / 'dataset' / 'tiles'

for folder in dataset_path.glob('fold_*'):
    target = folder / 'Faster'
    if target.exists():
        print(f"Removing {target}")
        for item in target.glob('*'):
            if item.is_dir():
                for sub in item.iterdir():
                    sub.unlink()
                item.rmdir()
            else:
                item.unlink()
        target.rmdir()
