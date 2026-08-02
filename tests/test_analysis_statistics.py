import pandas as pd
import pytest

from analysis.statistics import (
    descriptive_statistics,
    fold_completeness,
    model_ranking,
)


def sample_results():
    return pd.DataFrame(
        {
            "model_id": ["a", "a", "b", "b"],
            "detector": ["x"] * 4,
            "architecture": ["a", "a", "b", "b"],
            "fold": [1, 2, 1, 2],
            "mAP": [0.4, 0.6, 0.7, 0.8],
            "MAE": [4.0, 3.0, 2.0, 1.0],
        }
    )


def test_statistics_include_median_and_iqr():
    result = descriptive_statistics(sample_results())
    row = result[(result.model_id == "a") & (result.metric == "mAP")].iloc[0]

    assert row["count"] == 2
    assert row["mean"] == 0.5
    assert row["median"] == 0.5
    assert row["iqr"] == pytest.approx(0.1)


def test_statistics_ignore_missing_values_and_unknown_columns():
    frame = sample_results()
    frame.loc[1, "mAP"] = None
    frame["not_a_metric"] = [1, 2, 3, 4]

    result = descriptive_statistics(frame)
    row = result[(result.model_id == "a") & (result.metric == "mAP")].iloc[0]

    assert row["count"] == 1
    assert "not_a_metric" not in set(result["metric"])


def test_error_metric_is_ranked_lower_is_better():
    result = model_ranking(sample_results())
    mae = result[result.metric == "MAE"].sort_values("rank")

    assert mae.iloc[0].model_id == "b"


def test_overall_score_uses_normalized_metric_directions():
    result = model_ranking(sample_results())
    overall = result[result.metric == "overall"].set_index("model_id")

    assert overall.loc["b", "overall_score"] > overall.loc["a", "overall_score"]
    assert overall.loc["b", "rank"] == 1


def test_fold_completeness_reports_missing_folds():
    frame = sample_results().drop(index=1)

    result = fold_completeness(frame, expected_folds=2).set_index("model_id")

    assert result.loc["a", "observed_folds"] == 1
    assert result.loc["a", "expected_folds"] == 2
    assert result.loc["a", "missing_folds"] == [2]
    assert not result.loc["a", "complete"]
    assert result.loc["b", "complete"]


def test_fold_completeness_infers_observed_fold_set_when_unspecified():
    result = fold_completeness(sample_results(), expected_folds=None)

    assert set(result["missing_folds"][0]) == set()
    assert result["complete"].all()
