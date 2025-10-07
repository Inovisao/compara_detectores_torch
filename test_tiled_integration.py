#!/usr/bin/env python3
"""
Test script to verify tiled dataset integration

This script tests that the modified detector scripts can correctly:
1. Detect tiled dataset structure
2. Read annotations from fold directories
3. Copy images from the correct locations
"""

import os
import sys

# Set environment variable for tiled dataset mode
os.environ['USE_TILED_DATASET'] = 'true'

# Add src to path
sys.path.insert(0, 'src')

def test_yolov8():
    """Test YOLOV8 dataset generation with tiled data"""
    print("\n" + "="*80)
    print("Testing YOLOV8 with tiled dataset...")
    print("="*80)

    from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8

    fold = 'fold_1'
    root_data_dir = os.path.join('dataset', 'tiles', 'grid', fold)

    if not os.path.exists(root_data_dir):
        print(f"❌ Tiled dataset not found at {root_data_dir}")
        return False

    try:
        CriarLabelsYOLOV8(fold, root_data_dir)

        # Verify output
        yolo_dir = os.path.join(root_data_dir, 'YOLO')
        if not os.path.exists(yolo_dir):
            print(f"❌ YOLO output directory not created at {yolo_dir}")
            return False

        # Check splits
        for split in ['train', 'valid', 'test']:
            split_images = os.path.join(yolo_dir, split, 'images')
            split_labels = os.path.join(yolo_dir, split, 'labels')

            if not os.path.exists(split_images):
                print(f"❌ {split} images directory not created")
                return False

            if not os.path.exists(split_labels):
                print(f"❌ {split} labels directory not created")
                return False

            num_images = len([f for f in os.listdir(split_images) if f.endswith('.jpg')])
            num_labels = len([f for f in os.listdir(split_labels) if f.endswith('.txt')])

            print(f"✅ {split}: {num_images} images, {num_labels} labels")

        print("\n✅ YOLOV8 test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faster_rcnn():
    """Test Faster R-CNN dataset generation with tiled data"""
    print("\n" + "="*80)
    print("Testing Faster R-CNN with tiled dataset...")
    print("="*80)

    from Detectors.FasterRCNN.geradataset import geredata

    fold = 'fold_1'
    root_data_dir = os.path.join('dataset', 'tiles', 'grid', fold)

    if not os.path.exists(root_data_dir):
        print(f"❌ Tiled dataset not found at {root_data_dir}")
        return False

    try:
        geredata(fold, root_data_dir)

        # Verify output
        faster_dir = os.path.join(root_data_dir, 'Faster')
        if not os.path.exists(faster_dir):
            print(f"❌ Faster output directory not created at {faster_dir}")
            return False

        # Check splits
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(faster_dir, split)

            if not os.path.exists(split_dir):
                print(f"❌ {split} directory not created")
                return False

            annotations = os.path.join(split_dir, '_annotations.coco.json')
            if not os.path.exists(annotations):
                print(f"❌ {split} annotations not found")
                return False

            num_images = len([f for f in os.listdir(split_dir) if f.endswith('.jpg')])
            print(f"✅ {split}: {num_images} images, annotations present")

        print("\n✅ Faster R-CNN test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yolov5_tph():
    """Test YOLOV5_TPH dataset generation with tiled data"""
    print("\n" + "="*80)
    print("Testing YOLOV5_TPH with tiled dataset...")
    print("="*80)

    from Detectors.YOLOV5_TPH.GeraLabels import CriarLabelsYOLOV5TPH

    fold = 'fold_1'
    root_data_dir = os.path.join('dataset', 'tiles', 'grid', fold)

    if not os.path.exists(root_data_dir):
        print(f"❌ Tiled dataset not found at {root_data_dir}")
        return False

    try:
        CriarLabelsYOLOV5TPH(fold, root_data_dir)

        # Verify output
        yolo_dir = os.path.join(root_data_dir, 'YOLOV5_TPH')
        if not os.path.exists(yolo_dir):
            print(f"❌ YOLOV5_TPH output directory not created at {yolo_dir}")
            return False

        # Check splits
        for split in ['train', 'val', 'test']:
            split_images = os.path.join(yolo_dir, split, 'images')
            split_labels = os.path.join(yolo_dir, split, 'labels')

            if not os.path.exists(split_images):
                print(f"❌ {split} images directory not created")
                return False

            if not os.path.exists(split_labels):
                print(f"❌ {split} labels directory not created")
                return False

            num_images = len([f for f in os.listdir(split_images) if f.endswith('.jpg')])
            num_labels = len([f for f in os.listdir(split_labels) if f.endswith('.txt')])

            print(f"✅ {split}: {num_images} images, {num_labels} labels")

        print("\n✅ YOLOV5_TPH test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Tiled Dataset Integration Test")
    print("="*80)
    print(f"USE_TILED_DATASET: {os.getenv('USE_TILED_DATASET')}")

    results = []

    # Test each detector
    results.append(("YOLOV8", test_yolov8()))
    results.append(("Faster R-CNN", test_faster_rcnn()))
    results.append(("YOLOV5_TPH", test_yolov5_tph()))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
