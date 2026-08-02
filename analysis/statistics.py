"""Descriptive statistics, model rankings, and fold coverage."""

from typing import Optional, Set

import pandas as pd

from analysis.loader import METRICS


_LOWER_IS_BETTER = {"MAE", "RMSE"}
_STAT_COLUMNS = [
    "count",
    "mean",
    "median",
    "std",
    "min",
    "q1",
    "q3",
    "max",
    "iqr",
]


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [metric for metric in METRICS if metric in frame.columns]
    if not metric_columns:
        return pd.DataFrame(columns=["model_id", "metric", "value"])

    values = frame[["model_id"] + metric_columns].melt(
        id_vars=["model_id"], var_name="metric", value_name="value"
    )
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    return values.dropna(subset=["value"])


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return fold-level descriptive statistics for each model and metric."""
    values = _metric_frame(frame)
    columns = ["model_id", "metric"] + _STAT_COLUMNS
    if values.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (model_id, metric), group in values.groupby(["model_id", "metric"], sort=False):
        series = group["value"]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        rows.append(
            {
                "model_id": model_id,
                "metric": metric,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "q1": q1,
                "q3": q3,
                "max": series.max(),
                "iqr": q3 - q1,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def model_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank model means and provide an overall direction-normalized score."""
    values = _metric_frame(frame)
    columns = ["model_id", "metric", "mean", "rank", "normalized_score", "overall_score"]
    if values.empty:
        return pd.DataFrame(columns=columns)

    means = (
        values.groupby(["model_id", "metric"], sort=False)["value"]
        .mean()
        .reset_index(name="mean")
    )
    rows = []
    normalized_by_model = {}
    for metric, group in means.groupby("metric", sort=False):
        low = group["mean"].min()
        high = group["mean"].max()
        if high == low:
            normalized = pd.Series(1.0, index=group.index)
        elif metric in _LOWER_IS_BETTER:
            normalized = (high - group["mean"]) / (high - low)
        else:
            normalized = (group["mean"] - low) / (high - low)

        ranks = group["mean"].rank(
            ascending=metric in _LOWER_IS_BETTER, method="min"
        )
        for index, row in group.iterrows():
            score = float(normalized.loc[index])
            model_id = row["model_id"]
            normalized_by_model.setdefault(model_id, []).append(score)
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "mean": row["mean"],
                    "rank": ranks.loc[index],
                    "normalized_score": score,
                }
            )

    overall_rows = []
    for model_id, scores in normalized_by_model.items():
        overall_rows.append(
            {
                "model_id": model_id,
                "metric": "overall",
                "mean": pd.NA,
                "rank": pd.NA,
                "normalized_score": pd.NA,
                "overall_score": sum(scores) / len(scores),
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall["rank"] = overall["overall_score"].rank(ascending=False, method="min")
    result = pd.DataFrame(rows)
    result["overall_score"] = result["model_id"].map(
        overall.set_index("model_id")["overall_score"]
    )
    return pd.concat([result[columns], overall[columns]], ignore_index=True)


def _fold_key(value) -> object:
    text = str(value)
    if text.startswith("fold_"):
        suffix = text[5:]
        if suffix.isdigit():
            return int(suffix)
    return value


def fold_completeness(
    frame: pd.DataFrame, expected_folds: Optional[int]
) -> pd.DataFrame:
    """Report observed and missing fold identifiers for every model."""
    observed_by_model = {}
    all_observed: Set[object] = set()
    for model_id, group in frame.groupby("model_id", sort=False):
        folds = {_fold_key(value) for value in group["fold"].dropna().unique()}
        observed_by_model[model_id] = folds
        all_observed.update(folds)

    expected = (
        set(range(1, expected_folds + 1))
        if expected_folds is not None
        else set(all_observed)
    )
    expected_count = expected_folds if expected_folds is not None else len(expected)
    rows = []
    for model_id, observed in observed_by_model.items():
        missing = sorted(expected - observed, key=str)
        rows.append(
            {
                "model_id": model_id,
                "observed_folds": len(observed),
                "expected_folds": expected_count,
                "missing_folds": missing,
                "complete": not missing,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "observed_folds",
            "expected_folds",
            "missing_folds",
            "complete",
        ],
    )
