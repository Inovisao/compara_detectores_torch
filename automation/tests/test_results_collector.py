import pytest
from automation.results_collector import ResultsCollector
from automation.experiment_grid import Experiment

def test_results_collector_write_csv(tmp_path):
    """Test writing results to CSV."""
    csv_file = tmp_path / "results.csv"
    collector = ResultsCollector(str(csv_file))
    
    exp = Experiment(
        experiment_id="test_exp",
        model="YOLOV8",
        architecture="yolov8s",
        learning_rate=0.001,
        optimizer="AdamW",
        batch_size=32,
        weight_decay=0.0005,
        scheduler="cosine",
        epochs=10,
        patience=5,
        augmentations={"mosaic": 1.0},
        fold=1,
        seed=42
    )
    
    metrics = {
        "mAP": 0.85,
        "mAP50": 0.92,
        "mAP75": 0.78,
        "precision": 0.88,
        "recall": 0.82,
        "f1_score": 0.85
    }
    
    collector.write_result(exp, "model.pt", metrics, training_time=123.4)
    
    # Verify CSV was created
    assert csv_file.exists()
    
    # Read and verify content
    import csv
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['experiment_id'] == "test_exp"
        assert rows[0]['mAP'] == "0.85"
        assert rows[0]['training_time_s'] == "123.4"
