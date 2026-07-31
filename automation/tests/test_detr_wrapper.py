import pytest
from unittest.mock import Mock, patch
from automation.wrappers.detr_wrapper import DETRWrapper
from automation.experiment_grid import Experiment

def test_detr_wrapper_train():
    """Test DETR wrapper calls training correctly."""
    wrapper = DETRWrapper()
    
    exp = Experiment(
        experiment_id="Detr_detr_resnet50_lr0.0001_AdamW_bs4_multi_step_fold1_seed42",
        model="Detr",
        architecture="detr_resnet50",
        learning_rate=0.0001,
        optimizer="AdamW",
        batch_size=4,
        weight_decay=0.0001,
        scheduler="multi_step",
        epochs=50,
        patience=10,
        augmentations={},
        fold=1,
        seed=42
    )
    
    with patch('automation.wrappers.detr_wrapper.convert_coco_to_voc') as mock_convert, \
         patch('automation.wrappers.detr_wrapper.subprocess.run') as mock_run:
        
        model_path, metrics = wrapper.train(exp)
        
        mock_convert.assert_called_once_with(1)
        mock_run.assert_called_once()
        
        # Verify subprocess command
        cmd = mock_run.call_args[0][0]
        assert '--epochs' in cmd
        assert '50' in cmd
        assert '--learning-rate' in cmd
        assert '0.0001' in cmd
        
        assert model_path == "Detr/training/best_model.pth"
