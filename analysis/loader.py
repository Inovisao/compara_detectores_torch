"""Load result summaries from the supported result formats."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


METRICS = [
    "mAP",
    "mAP50",
    "mAP75",
    "MAE",
    "RMSE",
    "precision",
    "recall",
    "f1",
    "pearson_r",
]


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    legacy_detector = "detector" not in result and "ml" in result
    if "detector" not in result and "ml" in result:
        result["detector"] = result["ml"]
    if "architecture" not in result:
        result["architecture"] = result.get("detector", "unknown")
    if "fold" not in result:
        result["fold"] = "unknown"
    if legacy_detector:
        result["model_id"] = result["detector"].astype(str)
    else:
        result["model_id"] = (
            result["detector"].astype(str) + "/" + result["architecture"].astype(str)
        )
    for metric in METRICS:
        if metric in result:
            result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result


def _metadata_from_path(relative_path: Path) -> Dict[str, str]:
    parts = relative_path.parts[:-1]
    fold_index = next(
        (index for index, part in enumerate(parts) if part.startswith("fold_")),
        None,
    )
    if fold_index is None:
        return {"detector": "unknown", "architecture": "unknown", "fold": "unknown"}

    fold = parts[fold_index]
    detector = parts[fold_index - 1] if fold_index else "unknown"
    architecture = parts[fold_index + 1] if fold_index + 1 < len(parts) else detector
    return {"detector": detector, "architecture": architecture, "fold": fold}


def _load_json_results(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.rglob("metrics.json")):
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json must contain a JSON object: %s" % path)
        row = _metadata_from_path(path.relative_to(results_dir))
        row.update(metrics)
        rows.append(row)
    return normalize_results(pd.DataFrame(rows))


def load_results(results_dir: Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    if summary_path.is_file():
        return normalize_results(pd.read_csv(summary_path))

    if any(results_dir.rglob("metrics.json")):
        return _load_json_results(results_dir)

    raise FileNotFoundError(
        "No summary.csv or metrics.json found in results directory: %s" % results_dir
    )
