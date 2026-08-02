from pathlib import Path

import pandas as pd

from analysis.loader import load_results, normalize_results


def test_normalizes_legacy_ml_column():
    frame = pd.DataFrame({"ml": ["YOLOV8"], "fold": ["fold_1"], "mAP": [0.5]})
    result = normalize_results(frame)
    assert result.loc[0, "detector"] == "YOLOV8"
    assert result.loc[0, "model_id"] == "YOLOV8"


def test_loads_summary_csv_before_json(tmp_path: Path):
    pd.DataFrame(
        {"detector": ["yolo"], "architecture": ["yolov8s"], "fold": [1], "mAP": [0.5]}
    ).to_csv(tmp_path / "summary.csv", index=False)
    result = load_results(tmp_path)
    assert len(result) == 1
    assert result.loc[0, "mAP"] == 0.5


def test_loads_nested_metrics_json_when_csv_missing(tmp_path: Path):
    run = tmp_path / "fold_1" / "yolov8s"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"mAP": 0.6, "MAE": 2.0}')
    result = load_results(tmp_path)
    assert result.loc[0, "mAP"] == 0.6
    assert result.loc[0, "fold"] == "fold_1"
