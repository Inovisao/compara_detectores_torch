from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Detectors.YOLO26.GeraLabels import CriarLabelsYOLO26
from Detectors.YOLO26.config import finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINETUNE_DATASET = PROJECT_ROOT / "dataset" / "fine_tuning"
WEIGHTS = PROJECT_ROOT / "src" / "model_checkpoints" / "fold_1" / "YOLO26" / "train" / "weights" / "best.pt"

if not WEIGHTS.exists():
    raise FileNotFoundError(f"Pesos não encontrados: {WEIGHTS}")

data_yaml = CriarLabelsYOLO26("fold_1", FINETUNE_DATASET)
finetune(data_yaml, WEIGHTS)
