from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from analysis.pipeline import run_analysis
from analysis.plots import create_plots
from analysis.report import write_report
from cli import app


def test_analyze_command_is_registered():
    result = CliRunner().invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--results" in result.stdout


def _write_summary(results_dir: Path) -> None:
    pd.DataFrame(
        {
            "detector": ["yolo", "yolo", "faster", "faster"],
            "architecture": ["s", "s", "r", "r"],
            "fold": [1, 2, 1, 2],
            "mAP": [0.4, 0.6, 0.7, 0.8],
            "MAE": [4.0, 3.0, 2.0, 1.0],
        }
    ).to_csv(results_dir / "summary.csv", index=False)


def test_run_analysis_creates_outputs_and_propagates_expected_folds(tmp_path: Path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "analysis"
    results_dir.mkdir()
    _write_summary(results_dir)

    paths = run_analysis(results_dir, output_dir, expected_folds=3)

    for name in (
        "normalized_results",
        "descriptive_statistics",
        "model_ranking",
        "fold_completeness",
        "findings",
    ):
        assert paths[name].is_file()
    assert (output_dir / "charts" / "boxplot_map.png").is_file()
    completeness = pd.read_csv(paths["fold_completeness"])
    assert set(completeness["expected_folds"]) == {3}
    assert set(completeness["missing_folds"]) == {"[3]"}


@pytest.mark.parametrize(
    "columns",
    [
        {"detector": ["yolo"], "fold": [1], "other": ["value"]},
        {"detector": ["yolo"], "fold": [1], "mAP": ["invalid"]},
    ],
)
def test_run_analysis_rejects_results_without_usable_metrics(tmp_path: Path, columns):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "analysis"
    results_dir.mkdir()
    pd.DataFrame(columns).to_csv(results_dir / "summary.csv", index=False)

    with pytest.raises(ValueError, match="No usable metric data"):
        run_analysis(results_dir, output_dir)

    assert not output_dir.exists()


def test_analyze_cli_reports_input_errors(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame({"detector": ["yolo"], "mAP": ["invalid"]}).to_csv(
        results_dir / "summary.csv", index=False
    )

    result = CliRunner().invoke(
        app, ["analyze", "--results", str(results_dir), "--output", str(tmp_path / "analysis")]
    )

    assert result.exit_code != 0
    assert "Analysis failed" in result.output
    assert "No usable metric data" in result.output


@pytest.mark.parametrize("output_name", ["results", "results/analysis"])
def test_run_analysis_rejects_output_inside_raw_results(tmp_path: Path, output_name: str):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_summary(results_dir)

    with pytest.raises(ValueError, match="output directory.*results directory"):
        run_analysis(results_dir, tmp_path / output_name)


def test_outputs_charts_and_report(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "model_id": ["a", "a", "b", "b"],
            "fold": [1, 2, 1, 2],
            "mAP": [0.4, 0.6, 0.7, 0.8],
            "MAE": [4.0, 3.0, 2.0, 1.0],
            "precision": [0.5, 0.6, 0.7, 0.8],
            "recall": [0.6, 0.7, 0.8, 0.9],
            "f1": [0.55, 0.65, 0.75, 0.85],
        }
    )
    charts = create_plots(frame, tmp_path / "charts")

    assert any(path.name == "boxplot_map.png" for path in charts)
    assert any(path.name == "boxplot_mae.png" for path in charts)
    assert any(path.name == "metric_means.png" for path in charts)
    assert any(path.name == "classification_metrics.png" for path in charts)
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
    assert "mAP50" in content
    assert "mAP50 column is absent" in content
    assert "](/" not in content
    assert "](charts/boxplot_map.png)" in content


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
