#!/bin/bash
set -e

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT_DIR"

# Executa o arquivo treino.py
python src/Detectors/YOLOV8/config.py