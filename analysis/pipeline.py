"""Orchestrate the complete detector-results analysis."""

from pathlib import Path
from typing import Dict, Optional

from analysis.loader import load_results
from analysis.plots import create_plots
from analysis.report import write_report
from analysis.statistics import descriptive_statistics, fold_completeness, model_ranking


def run_analysis(
    results_dir: Path,
    output_dir: Path,
    expected_folds: Optional[int] = None,
) -> Dict[str, Path]:
    """Load results and write analysis tables, charts, and findings."""
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    try:
        output_dir.relative_to(results_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "output directory must not be equal to or inside results directory"
        )

    frame = load_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = descriptive_statistics(frame)
    rankings = model_ranking(frame)
    completeness = fold_completeness(frame, expected_folds)

    paths = {
        "normalized_results": output_dir / "normalized_results.csv",
        "descriptive_statistics": output_dir / "descriptive_statistics.csv",
        "model_ranking": output_dir / "model_ranking.csv",
        "fold_completeness": output_dir / "fold_completeness.csv",
    }
    frame.to_csv(paths["normalized_results"], index=False)
    stats.to_csv(paths["descriptive_statistics"], index=False)
    rankings.to_csv(paths["model_ranking"], index=False)
    completeness.to_csv(paths["fold_completeness"], index=False)

    chart_paths = create_plots(frame, output_dir / "charts")
    paths["findings"] = output_dir / "findings.md"
    write_report(frame, stats, rankings, completeness, chart_paths, paths["findings"])
    paths.update({"chart_%s" % path.stem: path for path in chart_paths})
    return paths
