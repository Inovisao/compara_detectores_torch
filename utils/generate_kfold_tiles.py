#!/usr/bin/env python3
"""
K-Fold Grid Tiling Driver Script

This script orchestrates the generation of tiled datasets for k-fold cross-validation.
It processes 6 folds × 3 splits (train/val/test) using grid-based tiling.

Requirements:
- Images must be in ./dataset/all/train/
- Fold JSON files must be in ./dataset/all/filesJSON/
- The create_dataset toolkit must be available

Configuration:
- Grid: 6 rows × 7 columns
- Min object coverage: 0.3
- Train/val: tiles without annotations are discarded
- Test: all tiles are kept (including empty ones)

Output:
- Tiled images and annotations saved to ./dataset/tiles/grid/fold_K/SPLIT/
- Metadata and summary reports generated for each split
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess

# Add create_dataset to Python path
REPO_ROOT = Path(__file__).parent.parent
CREATE_DATASET_PATH = REPO_ROOT / "create_dataset"
sys.path.insert(0, str(CREATE_DATASET_PATH))

from src.config.settings import AppConfig, TilingConfig, DatasetConfig, ProcessingConfig
from src.services.dataset.processor import DatasetProcessor
from src.models.coco import CocoDataset


# Configuration
NUM_FOLDS = 6
SPLITS = ["train", "val", "test"]
GRID_ROWS = 6
GRID_COLS = 7
MIN_OBJECT_COVERAGE = 0.3
SOURCE_IMAGES_DIR = REPO_ROOT / "dataset" / "all" / "train"
FOLD_JSON_DIR = REPO_ROOT / "dataset" / "all" / "filesJSON"
OUTPUT_ROOT = REPO_ROOT / "dataset" / "tiles" / "grid"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def generate_metadata(fold: int, split: str, config: TilingConfig,
                      annotations_path: str) -> Dict:
    """Generate metadata for a processed split."""
    return {
        "fold": fold,
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "source_annotations": str(annotations_path),
        "source_images_dir": str(SOURCE_IMAGES_DIR),
        "tiling_config": {
            "grid_mode": config.grid_mode,
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "min_object_coverage": config.min_object_coverage,
            "keep_empty_tiles": config.keep_empty_tiles,
            "resize_output": config.resize_output,
        }
    }


def generate_summary_report(output_dir: Path, split: str) -> Dict:
    """Generate summary report for a processed split."""
    annotations_path = output_dir / "_annotations.coco.json"

    if not annotations_path.exists():
        return {"error": "Annotations file not found"}

    try:
        dataset = CocoDataset.from_json(str(annotations_path))

        # Count annotations per category
        category_counts = {}
        for ann in dataset.annotations:
            cat_id = ann.category_id
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        # Get category names
        category_map = {cat.id: cat.name for cat in dataset.categories}
        category_stats = {
            category_map.get(cat_id, f"category_{cat_id}"): count
            for cat_id, count in category_counts.items()
        }

        summary = {
            "split": split,
            "num_images": len(dataset.images),
            "num_annotations": len(dataset.annotations),
            "num_categories": len(dataset.categories),
            "category_counts": category_stats,
            "images_without_annotations": sum(
                1 for img in dataset.images
                if not any(ann.image_id == img.id for ann in dataset.annotations)
            )
        }

        return summary
    except Exception as e:
        return {"error": str(e)}


def process_fold_split(fold: int, split: str) -> bool:
    """Process a single fold-split combination.

    Args:
        fold: Fold number (1-6)
        split: Split name (train/val/test)

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"🔄 Processing Fold {fold} - {split.upper()}")
    print("=" * 80)

    # Paths
    annotations_path = FOLD_JSON_DIR / f"fold_{fold}_{split}.json"
    output_dir = OUTPUT_ROOT / f"fold_{fold}" / split

    # Validate input
    if not annotations_path.exists():
        print(f"❌ Error: Annotations file not found: {annotations_path}")
        return False

    if not SOURCE_IMAGES_DIR.exists():
        print(f"❌ Error: Source images directory not found: {SOURCE_IMAGES_DIR}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure tiling
    keep_empty = (split == "test")  # Only keep empty tiles for test split

    tiling_config = TilingConfig(
        grid_mode=True,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        min_object_coverage=MIN_OBJECT_COVERAGE,
        keep_empty_tiles=keep_empty,
        resize_output=None,  # No resizing
        tile_size=(512, 512),  # Not used in grid mode
        overlap=0,  # Not used in grid mode
    )

    dataset_config = DatasetConfig(
        input_path=str(SOURCE_IMAGES_DIR.parent),
        output_path=str(output_dir.parent),
    )

    processing_config = ProcessingConfig()

    app_config = AppConfig(
        tiling=tiling_config,
        dataset=dataset_config,
        processing=processing_config
    )

    # Process the dataset
    try:
        processor = DatasetProcessor(app_config)
        processor.process_dataset(
            annotations_path=str(annotations_path),
            images_dir=str(SOURCE_IMAGES_DIR),
            split_name=split
        )

        # Validate output
        print("\n🔍 Validating output...")
        if not processor.validate_output(split_name=split):
            print("❌ Validation failed")
            return False
        print("✅ Validation successful")

        # Generate metadata
        print("📝 Generating metadata...")
        metadata = generate_metadata(fold, split, tiling_config, annotations_path)
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   Saved to: {metadata_path}")

        # Generate summary report
        print("📊 Generating summary report...")
        summary = generate_summary_report(output_dir, split)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Also save as text for easy reading
        summary_txt_path = output_dir / "summary.txt"
        with open(summary_txt_path, "w") as f:
            f.write(f"Summary Report - Fold {fold} - {split.upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if "error" in summary:
                f.write(f"Error: {summary['error']}\n")
            else:
                f.write(f"Number of images: {summary['num_images']}\n")
                f.write(f"Number of annotations: {summary['num_annotations']}\n")
                f.write(f"Images without annotations: {summary['images_without_annotations']}\n")
                f.write(f"Number of categories: {summary['num_categories']}\n\n")
                f.write("Annotations per category:\n")
                for cat_name, count in summary['category_counts'].items():
                    f.write(f"  - {cat_name}: {count}\n")

        print(f"   Saved to: {summary_path}")
        print(f"   Text report: {summary_txt_path}")

        print(f"\n✅ Fold {fold} - {split.upper()} completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error processing fold {fold} - {split}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main orchestration function."""
    print("=" * 80)
    print("K-Fold Grid Tiling Pipeline")
    print("=" * 80)
    print(f"Folds: {NUM_FOLDS}")
    print(f"Splits per fold: {', '.join(SPLITS)}")
    print(f"Grid: {GRID_ROWS} rows × {GRID_COLS} columns")
    print(f"Min object coverage: {MIN_OBJECT_COVERAGE}")
    print(f"Source images: {SOURCE_IMAGES_DIR}")
    print(f"Fold JSONs: {FOLD_JSON_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 80)

    # Confirm before starting
    response = input("\nProceed with tiling? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return

    start_time = time.time()
    results = []

    # Process each fold and split
    for fold in range(1, NUM_FOLDS + 1):
        for split in SPLITS:
            success = process_fold_split(fold, split)
            results.append({
                "fold": fold,
                "split": split,
                "success": success
            })

    # Summary
    elapsed = time.time() - start_time
    elapsed_mins = int(elapsed // 60)
    elapsed_secs = int(elapsed % 60)

    print("\n" + "=" * 80)
    print("🎉 K-Fold Tiling Pipeline Complete!")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nResults: {successful}/{total} successful")
    print(f"Time elapsed: {elapsed_mins}m {elapsed_secs}s")

    if successful < total:
        print("\n⚠️  Some splits failed:")
        for r in results:
            if not r["success"]:
                print(f"   - Fold {r['fold']} - {r['split']}")
        sys.exit(1)
    else:
        print("\n✅ All folds and splits processed successfully!")
        print(f"\nOutput directory: {OUTPUT_ROOT}")
        print("\nNext steps:")
        print("1. Review the summary reports in each fold/split directory")
        print("2. Verify tile counts and annotation distributions")
        print("3. Update your training configuration to use the tiled datasets")
        sys.exit(0)


if __name__ == "__main__":
    main()
