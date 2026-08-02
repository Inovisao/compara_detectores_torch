from pathlib import Path

import pandas as pd

from analysis.plots import create_plots
from analysis.report import write_report


def test_outputs_charts_and_report(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "model_id": ["a", "a", "b", "b"],
            "fold": [1, 2, 1, 2],
            "mAP": [0.4, 0.6, 0.7, 0.8],
            "MAE": [4.0, 3.0, 2.0, 1.0],
        }
    )
    charts = create_plots(frame, tmp_path / "charts")

    assert any(path.name == "boxplot_map.png" for path in charts)
    assert any(path.name == "boxplot_mae.png" for path in charts)
    assert any(path.name == "metric_means.png" for path in charts)
    assert all(path.is_file() for path in charts)

    report = tmp_path / "findings.md"
    write_report(frame, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), charts, report)
    content = report.read_text()
    assert "# Detector Results Analysis" in content
    assert "## Dataset Coverage" in content
    assert "## Best Models Per Metric" in content
    assert "## Overall Ranking" in content
    assert "## Stability" in content
    assert "## Warnings" in content
    assert "## Charts" in content
    assert "higher is better" in content
    assert "lower is better" in content


def test_outputs_skip_empty_metrics_and_warn_about_missing_data(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "model_id": ["a", "a"],
            "fold": [1, 2],
            "mAP": [0.4, None],
            "MAE": [None, None],
        }
    )
    charts = create_plots(frame, tmp_path / "charts")

    assert not (tmp_path / "charts" / "boxplot_mae.png").exists()
    assert (tmp_path / "charts" / "boxplot_map.png").exists()

    report = tmp_path / "findings.md"
    write_report(
        frame,
        pd.DataFrame(
            {
                "model_id": ["a", "a"],
                "metric": ["mAP", "MAE"],
                "count": [1, 0],
                "mean": [0.4, None],
                "median": [0.4, None],
                "std": [None, None],
                "min": [0.4, None],
                "q1": [0.4, None],
                "q3": [0.4, None],
                "max": [0.4, None],
                "iqr": [0.0, None],
            }
        ),
        pd.DataFrame(),
        pd.DataFrame(),
        charts,
        report,
    )
    content = report.read_text()
    assert "missing" in content.lower()
    assert "MAE" in content
