import pytest
from automation.wrappers.base_wrapper import BaseWrapper
from automation.experiment_grid import Experiment

def test_base_wrapper_abstract():
    """BaseWrapper should be abstract and not instantiable."""
    with pytest.raises(TypeError):
        BaseWrapper()

def test_concrete_wrapper():
    """Concrete implementation should work."""
    class TestWrapper(BaseWrapper):
        def train(self, experiment: Experiment) -> tuple[str, dict]:
            return "model.pt", {"mAP": 0.85}
    
    wrapper = TestWrapper()
    exp = Experiment(
        experiment_id="test",
        model="Test",
        architecture="test_arch",
        learning_rate=0.001,
        optimizer="AdamW",
        batch_size=32,
        weight_decay=0.0005,
        scheduler="cosine",
        epochs=10,
        patience=5,
        augmentations={},
        fold=1,
        seed=42
    )
    
    model_path, metrics = wrapper.train(exp)
    assert model_path == "model.pt"
    assert metrics["mAP"] == 0.85
