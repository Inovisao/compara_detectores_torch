import pytest
from unittest.mock import Mock, patch
from automation.wrappers.yolov8_wrapper import YOLOV8Wrapper
from automation.experiment_grid import Experiment

def test_yolov8_wrapper_train():
    """Test YOLOV8 wrapper calls training correctly."""
    wrapper = YOLOV8Wrapper()
    
    exp = Experiment(
        experiment_id="YOLOV8_yolov8s_lr0.001_AdamW_bs32_cosine_fold1_seed42",
        model="YOLOV8",
        architecture="yolov8s",
        learning_rate=0.001,
        optimizer="AdamW",
        batch_size=32,
        weight_decay=0.0005,
        scheduler="cosine",
        epochs=10,
        patience=5,
        augmentations={"mosaic": 1.0, "flipud": 0.5},
        fold=1,
        seed=42
    )
    
    with patch('automation.wrappers.yolov8_wrapper.YOLO') as mock_yolo, \
         patch('automation.wrappers.yolov8_wrapper.CriarLabelsYOLOV8') as mock_labels, \
         patch('automation.wrappers.yolov8_wrapper.Settings') as mock_settings:
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        model_path, metrics = wrapper.train(exp)
        
        # Verify calls
        mock_settings.assert_called_once()
        mock_labels.assert_called_once_with(1)
        mock_yolo.assert_called_once_with("yolov8s.pt")
        
        # Verify train was called with correct params
        train_call = mock_model.train.call_args
        assert train_call.kwargs['lr0'] == 0.001
        assert train_call.kwargs['optimizer'] == "AdamW"
        assert train_call.kwargs['batch'] == 32
        assert train_call.kwargs['mosaic'] == 1.0
        assert train_call.kwargs['flipud'] == 0.5
        
        assert model_path == "YOLOV8/train/weights/best.pt"
