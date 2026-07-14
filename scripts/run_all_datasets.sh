# ── 7. Rodar all: treino do zero + avaliação ──────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Dataset: all"
echo " DATASET_ROOT: $SLICE_DS/all"
echo " MODEL_CHECKPOINTS_ROOT: $PROJECT_ROOT/model_checkpoints"
echo "════════════════════════════════════════"

DATASET_ROOT="$SLICE_DS/all" \
MODEL_CHECKPOINTS_ROOT="$PROJECT_ROOT/model_checkpoints" \
python main.py

echo ""
echo "[done] asahi_rect e all concluídos."


#!/usr/bin/env bash
# Corrige estrutura de checkpoints e executa main.py para asahi_rect e all.
#
# Fixes aplicados:
#   asahi/fold_1/Detr   – arquivos direto em Detr/, sem subdir training/
#   asahi_rect          – folds direto em pesos/asahi_rect/ (sem model_checkpoints/)
#   asahi_rect (todos)  – 15 checkpoints (5×3) treinados no dataset errado
#                         (buracos/all_320); remove tudo e retreina do zero
#
# Uso: bash scripts/run_all_datasets.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PESOS="/home/neto/development/slice_inference_api/pesos"
SLICE_DS="/home/neto/development/slice_inference_api/dataset"

# ── 1. Fix asahi/fold_1/Detr: criar subdir training/ com symlink ──────────────
ASAHI_FOLD1_DETR="$PESOS/asahi/model_checkpoints/fold_1/Detr"
if [ -f "$ASAHI_FOLD1_DETR/best_model.pth" ] && [ ! -d "$ASAHI_FOLD1_DETR/training" ]; then
    echo "[fix] Criando $ASAHI_FOLD1_DETR/training/"
    mkdir -p "$ASAHI_FOLD1_DETR/training"
    ln -sf ../best_model.pth "$ASAHI_FOLD1_DETR/training/best_model.pth"
fi

# ── 2. Fix asahi_rect: criar model_checkpoints/ como symlink para . ───────────
ASAHI_RECT_DIR="$PESOS/asahi_rect"
ASAHI_RECT_CKPT="$ASAHI_RECT_DIR/model_checkpoints"
if [ ! -e "$ASAHI_RECT_CKPT" ]; then
    echo "[fix] Criando symlink $ASAHI_RECT_CKPT -> ."
    ln -sf . "$ASAHI_RECT_CKPT"
fi

# ── 3. Remove TODOS os checkpoints errados do asahi_rect ─────────────────────
# Os 15 checkpoints (5 folds × YOLO/Faster/Detr) foram treinados no dataset
# errado (buracos/all_320), provavelmente por cópia de pasta de outro experimento.
echo "[fix] Removendo todos os checkpoints incorretos do asahi_rect..."
for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
    for model in Detr Faster YOLOV8; do
        ckpt_dir="$ASAHI_RECT_DIR/$fold/$model"
        if [ -d "$ckpt_dir" ]; then
            echo "  removendo $ckpt_dir"
            rm -rf "$ckpt_dir"
        fi
    done
done

# ── 4. Garantir filesJSON/ para 'all' ─────────────────────────────────────────
ALL_FILES_JSON="$SLICE_DS/all/filesJSON"
if [ ! -d "$ALL_FILES_JSON" ]; then
    echo "[setup] Gerando filesJSON/ para dataset all..."
    python3 "$SCRIPT_DIR/gen_all_folds.py"
fi

# ── 5. Rodar asahi_rect: treino completo dos 3 modelos ───────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Dataset: asahi_rect  (YOLOV8 + Faster + Detr)"
echo " DATASET_ROOT: $SLICE_DS/asahi_rect"
echo " MODEL_CHECKPOINTS_ROOT: $ASAHI_RECT_CKPT"
echo "════════════════════════════════════════"

cd "$PROJECT_ROOT/src"

DATASET_ROOT="$SLICE_DS/asahi_rect" \
MODEL_CHECKPOINTS_ROOT="$ASAHI_RECT_CKPT" \
python main.py


