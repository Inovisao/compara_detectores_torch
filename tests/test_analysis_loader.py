from pathlib import Path

import pandas as pd
import pytest

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
    json_run = tmp_path / "json_detector" / "fold_9" / "json_arch"
    json_run.mkdir(parents=True)
    (json_run / "metrics.json").write_text('{"mAP": 0.9}')
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


def test_falls_back_to_json_for_unusable_summary_csv(tmp_path: Path):
    (tmp_path / "summary.csv").write_text("")
    run = tmp_path / "detector" / "fold_2" / "architecture"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"mAP": 0.7}')

    result = load_results(tmp_path)

    assert result.loc[0, "mAP"] == 0.7
    assert result.loc[0, "fold"] == "fold_2"


def test_falls_back_to_json_for_empty_summary_frame(tmp_path: Path):
    (tmp_path / "summary.csv").write_text("detector,architecture\n")
    run = tmp_path / "detector" / "fold_3" / "architecture"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"mAP": 0.8}')

    result = load_results(tmp_path)

    assert result.loc[0, "mAP"] == 0.8
    assert result.loc[0, "fold"] == "fold_3"


def test_falls_back_to_json_for_malformed_summary_csv(tmp_path: Path):
    (tmp_path / "summary.csv").write_text('"detector,architecture\n')
    run = tmp_path / "detector" / "fold_4" / "architecture"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"mAP": 0.9}')

    result = load_results(tmp_path)

    assert result.loc[0, "mAP"] == 0.9
    assert result.loc[0, "fold"] == "fold_4"


def test_normalizes_missing_identity_columns_to_unknown():
    result = normalize_results(pd.DataFrame({"mAP": ["0.5"]}))

    assert result.loc[0, "detector"] == "unknown"
    assert result.loc[0, "architecture"] == "unknown"
    assert result.loc[0, "fold"] == "unknown"
    assert result.loc[0, "model_id"] == "unknown/unknown"


def test_normalizes_modern_identity_columns():
    result = normalize_results(
        pd.DataFrame({"detector": ["yolo"], "architecture": ["yolov8s"]})
    )

    assert result.loc[0, "model_id"] == "yolo/yolov8s"


def test_coerces_recognized_metrics_to_numeric():
    result = normalize_results(
        pd.DataFrame({"detector": ["yolo"], "mAP": ["0.5"], "MAE": ["invalid"]})
    )

    assert result.loc[0, "mAP"] == 0.5
    assert pd.isna(result.loc[0, "MAE"])


def test_preserves_multiple_json_folds(tmp_path: Path):
    for fold, value in (("fold_1", 0.4), ("fold_2", 0.6)):
        run = tmp_path / "detector" / fold / "architecture"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text('{"mAP": %s}' % value)

    result = load_results(tmp_path)

    assert set(result["fold"]) == {"fold_1", "fold_2"}
    assert set(result["mAP"]) == {0.4, 0.6}


def test_raises_clear_error_when_no_usable_source(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No usable results source"):
        load_results(tmp_path)


def test_loading_does_not_modify_inputs(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text("detector,architecture,fold,mAP\nyolo,yolov8s,1,0.5\n")
    before = summary.read_bytes()
    before_mtime = summary.stat().st_mtime_ns

    load_results(tmp_path)

    assert summary.read_bytes() == before
    assert summary.stat().st_mtime_ns == before_mtime
