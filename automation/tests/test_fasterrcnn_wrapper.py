import pytest
from unittest.mock import Mock, patch
from automation.wrappers.fasterrcnn_wrapper import FasterRCNNWrapper
from automation.experiment_grid import Experiment

def test_fasterrcnn_wrapper_resnet():
    """Test FasterRCNN wrapper with ResNet backbone."""
    wrapper = FasterRCNNWrapper()
    
    exp = Experiment(
        experiment_id="Faster_resnet50_lr0.001_SGD_bs8_step_fold1_seed42",
        model="Faster",
        architecture="resnet50",
        learning_rate=0.001,
        optimizer="SGD",
        batch_size=8,
        weight_decay=0.0005,
        scheduler="step",
        epochs=10,
        patience=5,
        augmentations={"horizontal_flip": True},
        fold=1,
        seed=42
    )
    
    with patch('automation.wrappers.fasterrcnn_wrapper.geredata') as mock_data:
        model_path, metrics, arch_type = wrapper.train(exp)
        
        mock_data.assert_called_once_with(1)
        assert arch_type == "resnet"
        assert model_path == "Faster/best.pth"

def test_fasterrcnn_wrapper_swin():
    """Test FasterRCNN wrapper with Swin backbone."""
    wrapper = FasterRCNNWrapper()
    
    exp = Experiment(
        experiment_id="Faster_swin_tiny_lr0.001_AdamW_bs8_step_fold1_seed42",
        model="Faster",
        architecture="swin_tiny",
        learning_rate=0.001,
        optimizer="AdamW",
        batch_size=8,
        weight_decay=0.0005,
        scheduler="step",
        epochs=10,
        patience=5,
        augmentations={},
        fold=1,
        seed=42
    )
    
    with patch('automation.wrappers.fasterrcnn_wrapper.geredata') as mock_data, \
         patch.object(wrapper, '_build_swin_model', return_value=Mock()) as mock_build:
        model_path, metrics, arch_type = wrapper.train(exp)
        
        assert arch_type == "swin_backbone"
