"""English Markdown reporting for detector result analysis."""

from pathlib import Path
from typing import Iterable
import os

import pandas as pd

from analysis.loader import METRICS


_LOWER_IS_BETTER = {"MAE", "RMSE"}


def _value(value) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or "none"
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return "%.4f" % value
    return str(value)


def _rows(table: pd.DataFrame, columns: Iterable[str]):
    if table.empty:
        return ["No data available."]
    names = list(columns)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_value(row.get(column, pd.NA)) for column in names) + " |")
    return lines


def write_report(
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    rankings: pd.DataFrame,
    completeness: pd.DataFrame,
    chart_paths,
    output_path: Path,
) -> None:
    """Write a descriptive report without making statistical significance claims."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [metric for metric in METRICS if metric in frame.columns]
    lines = ["# Detector Results Analysis", "", "## Dataset Coverage", ""]
    lines.extend(
        [
            "- Rows: %d" % len(frame),
            "- Models: %d" % frame["model_id"].nunique() if "model_id" in frame else "- Models: N/A",
            "- Folds: %d" % frame["fold"].nunique() if "fold" in frame else "- Folds: N/A",
            "- Expected metrics: %s" % ", ".join(METRICS),
            "- Metrics found: %s" % (", ".join(metrics) if metrics else "none"),
            "",
            "Metric availability:",
        ]
    )
    lines.extend(
        "- %s: %s" % (metric, "present" if metric in frame.columns else "absent")
        for metric in METRICS
    )
    lines.extend(
        [
            "",
            "## Best Models Per Metric",
            "",
        ]
    )
    best = rankings[rankings["metric"].isin(metrics)].copy() if "metric" in rankings else pd.DataFrame()
    if not best.empty:
        best = best.sort_values(["metric", "rank"])[["metric", "model_id", "mean", "rank"]]
    lines.extend(_rows(best, ["metric", "model_id", "mean", "rank"]))
    lines.extend(["", "Metric direction: higher is better for mAP, mAP50, mAP75, precision, recall, f1, and pearson_r; lower is better for MAE and RMSE.", "", "## Overall Ranking", ""])
    overall = rankings[rankings["metric"] == "overall"] if "metric" in rankings else pd.DataFrame()
    lines.extend(_rows(overall, ["model_id", "overall_score", "rank"]))
    lines.extend(["", "## Stability", ""])
    stability = stats[["model_id", "metric", "count", "std", "iqr"]] if not stats.empty and all(column in stats for column in ["model_id", "metric", "count", "std", "iqr"]) else pd.DataFrame()
    lines.extend(_rows(stability, ["model_id", "metric", "count", "std", "iqr"]))
    lines.extend(["", "## Warnings", ""])
    warnings = []
    for metric in METRICS:
        if metric not in frame.columns:
            warnings.append("- %s column is absent from the input results." % metric)
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        missing = int(values.isna().sum())
        if missing:
            warnings.append("- %s has %d missing or non-numeric value(s)." % (metric, missing))
        if values.notna().sum() == 0:
            warnings.append("- %s has no usable data and was omitted from charts and comparisons." % metric)
    if not completeness.empty and "complete" in completeness:
        for _, row in completeness[~completeness["complete"].fillna(False)].iterrows():
            warnings.append("- Model %s is missing fold(s): %s." % (row["model_id"], _value(row.get("missing_folds"))))
    lines.extend(warnings or ["No missing-data warnings were generated."])
    lines.extend(["", "No statistical significance claims are made; results are descriptive and based on the available folds.", "", "## Charts", ""])
    for path in chart_paths:
        chart_path = Path(path)
        link = os.path.relpath(chart_path, output_path.parent) if chart_path.is_absolute() else chart_path.as_posix()
        lines.append("- [%s](%s)" % (chart_path.name, Path(link).as_posix()))
    if not chart_paths:
        lines.append("No charts were created because no metric had usable data.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
