import os
import sys
import numpy as np

# Get model and fold from environment
MODEL = os.getenv('MODELS_TO_RUN', 'YOLOV8')
FOLD_NUM = int(os.getenv('FOLD_NUM', '1'))

print(f"\nQuick Test Mode: Training {MODEL} on fold_{FOLD_NUM}\n")

# Import from main.py
sys.path.insert(0, os.path.dirname(__file__))
from main import train_model, test_model, create_csv, generate_results
from main import GeraRult, save_imgs, GeraResultByClass, APENAS_TESTE

# Configuration
fold = f'fold_{FOLD_NUM}'
fold_dir = os.path.join('model_checkpoints', fold)
current_root = os.path.join('..', 'dataset', 'tiles', 'grid', fold)

if not os.path.exists(current_root):
    print(f"Error: Tiled dataset not found at {current_root}")
    sys.exit(1)

print(f"Dataset root: {current_root}")
print(f"Model checkpoint: {fold_dir}")
print("")

# Train the model
if not APENAS_TESTE:
    print(f"Training {MODEL} on {fold}...")
    model_path = train_model(MODEL, fold, fold_dir, current_root)
    if model_path is None:
        print("Training skipped (already exists with CONTINUE=True)")
        model_path = test_model(MODEL, fold_dir)
else:
    model_path = test_model(MODEL, fold_dir)

print(f"\nModel saved to: {model_path}")

# Generate results if enabled
if GeraRult:
    print("\nGenerating results...")
    create_csv(root=current_root, fold=fold, selected_model=MODEL, model_path=model_path, save_imgs=save_imgs)

if GeraResultByClass:
    print("\nGenerating per-class results...")
    generate_results(root=current_root, fold=fold, model=model_path, model_name=MODEL, save_imgs=save_imgs)

print("\n✓ Quick test completed!")
