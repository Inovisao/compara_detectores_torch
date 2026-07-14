#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

for ds in sahi asahi asahi_rect; do
    echo "=== $ds ==="
    python3 "$SCRIPT_DIR/adapt_yolo_folds.py" --dataset "$PROJECT_ROOT/dataset/$ds" "$@"
    echo ""
done
