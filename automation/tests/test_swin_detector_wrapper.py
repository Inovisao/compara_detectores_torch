import pytest
from unittest.mock import Mock, patch
from automation.wrappers.swin_detector_wrapper import SwinDetectorWrapper
from automation.experiment_grid import Experiment

def test_swin_detector_wrapper_train():
    """Test SwinDetector wrapper."""
    wrapper = SwinDetectorWrapper()
    
    exp = Experiment(
        experiment_id="SwinDetector_swin_tiny_lr0.0001_AdamW_bs4_cosine_fold1_seed42",
        model="SwinDetector",
        architecture="swin_tiny",
        learning_rate=0.0001,
        optimizer="AdamW",
        batch_size=4,
        weight_decay=0.0001,
        scheduler="cosine",
        epochs=50,
        patience=10,
        augmentations={},
        fold=1,
        seed=42
    )
    
    with patch('automation.wrappers.swin_detector_wrapper.timm.create_model') as mock_create:
        mock_create.return_value = Mock()
        model_path, metrics = wrapper.train(exp)
    
    assert model_path == "SwinDetector/best_model.pth"
