#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_contract import validate_dataset_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida o contrato de dataset COCO tileado.")
    parser.add_argument("--root", required=True, type=Path, help="Raiz do dataset, ex.: dataset/asahi_rect")
    args = parser.parse_args()

    errors = validate_dataset_contract(args.root)
    if errors:
        print(f"[ERRO] Dataset inválido: {args.root}")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"[OK] Dataset válido: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
