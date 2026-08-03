#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="train-and-analyze"
if [[ "${1:-}" == "--analysis-only" ]]; then
  MODE="analysis-only"
  shift
fi

if [[ "$MODE" == "analysis-only" ]]; then
  RESULTS_DIR="${1:-results/faster_rcnn_continue}"
  ANALYSIS_DIR="${2:-results_prototype_analysis}"
else
  CONFIG_PATH="${1:-configs/sweeps/faster_rcnn_continue.yaml}"
  RESULTS_DIR="${2:-results/faster_rcnn_continue}"
  ANALYSIS_DIR="${3:-results_prototype_analysis}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$MODE" != "analysis-only" ]]; then
  echo "Starting training sweep: $CONFIG_PATH"
  "$PYTHON_BIN" cli.py sweep --config "$CONFIG_PATH"
fi

echo "Generating analysis: $RESULTS_DIR -> $ANALYSIS_DIR"
"$PYTHON_BIN" cli.py analyze --results "$RESULTS_DIR" --output "$ANALYSIS_DIR"

echo "Complete. Findings: $ANALYSIS_DIR/findings.md"
