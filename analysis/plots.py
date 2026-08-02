"""Non-interactive charts for detector result summaries."""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from analysis.loader import METRICS


def _usable_metrics(frame: pd.DataFrame):
    for metric in METRICS:
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if not values.empty:
            yield metric, values


def create_plots(frame: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create metric boxplots and a grouped metric-means chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    usable = []

    for metric in METRICS:
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        grouped = []
        labels = []
        for model_id, group in frame.assign(_value=values).groupby(
            "model_id", sort=False
        ):
            model_values = group["_value"].dropna()
            if not model_values.empty:
                labels.append(str(model_id))
                grouped.append(model_values.tolist())
        if not grouped:
            continue

        figure, axis = plt.subplots(figsize=(7, 4))
        axis.boxplot(grouped)
        axis.set_xticklabels(labels)
        axis.set_title("%s by model and fold" % metric)
        axis.set_xlabel("Model")
        axis.set_ylabel(metric)
        figure.tight_layout()
        path = output_dir / ("boxplot_%s.png" % metric.lower())
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)
        usable.append((metric, labels, [sum(values) / len(values) for values in grouped]))

    if usable:
        models = []
        for _, labels, _ in usable:
            for label in labels:
                if label not in models:
                    models.append(label)
        figure, axis = plt.subplots(figsize=(8, 4))
        width = 0.8 / len(usable)
        positions = list(range(len(models)))
        for index, (metric, labels, means) in enumerate(usable):
            by_model = dict(zip(labels, means))
            offsets = [position + index * width for position in positions]
            axis.bar(offsets, [by_model.get(model, float("nan")) for model in models], width, label=metric)
        axis.set_title("Mean metric values by model")
        axis.set_xlabel("Model")
        axis.set_ylabel("Mean value")
        axis.set_xticks([position + width * (len(usable) - 1) / 2 for position in positions])
        axis.set_xticklabels(models)
        axis.legend()
        figure.tight_layout()
        path = output_dir / "metric_means.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

    return created
