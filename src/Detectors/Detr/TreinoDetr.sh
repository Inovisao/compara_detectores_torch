#!/bin/bash
set -euo pipefail

echo "[TreinoDetr.sh] Iniciando treino DETR - $(date)"
echo "[TreinoDetr.sh] CWD=$(pwd)"
python Detectors/Detr/train_detector.py
echo "[TreinoDetr.sh] Treino DETR finalizado - $(date)"