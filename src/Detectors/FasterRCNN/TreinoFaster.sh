#!/bin/bash
set -euo pipefail

echo "[TreinoFaster.sh] Iniciando treino FasterRCNN - $(date)"
echo "[TreinoFaster.sh] CWD=$(pwd)"
echo "[TreinoFaster.sh] FASTER_TRAIN_DIR=${FASTER_TRAIN_DIR:-<não definido>}"
echo "[TreinoFaster.sh] FASTER_VAL_DIR=${FASTER_VAL_DIR:-<não definido>}"
python Detectors/FasterRCNN/train.py
echo "[TreinoFaster.sh] Treino FasterRCNN finalizado - $(date)"