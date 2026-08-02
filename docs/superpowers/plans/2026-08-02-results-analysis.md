# Results Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `cli.py analyze` command that generates fold-level statistics, boxplots, CSV summaries, and an English Markdown findings report from detector results.

**Architecture:** A focused `analysis/` package separates loading/normalization, statistics, plotting, and report generation. The CLI orchestrates these pure-ish components and writes only to a caller-selected output directory; raw result files remain unchanged.

**Tech Stack:** Python 3.9+, pandas, NumPy, Matplotlib, PyYAML-independent CLI integration, pytest-style tests.

## Global Constraints

- Each fold-level result is one statistical observation.
- Accept `summary.csv` first and nested `metrics.json` files as fallback.
- Normalize legacy `ml` columns and modern `detector`/`architecture` columns.
- Missing metrics remain missing and are reported, never silently imputed.
- `MAE` and `RMSE` are lower-is-better; all other ranking metrics are higher-is-better.
- The analysis is read-only with respect to raw result files.
- Use a non-interactive Matplotlib backend for server/automation execution.
- Generated findings are descriptive and must not claim statistical significance without sufficient fold observations.

## File Map

| File | Responsibility |
|------|----------------|
| `analysis/__init__.py` | Package marker |
| `analysis/loader.py` | Load and normalize CSV/JSON result data |
| `analysis/statistics.py` | Descriptive statistics, rankings, completeness, overall score |
| `analysis/plots.py` | Boxplots and mean comparison chart |
| `analysis/report.py` | English Markdown findings generation |
| `analysis/pipeline.py` | End-to-end analysis orchestration |
| `cli.py` | Add `analyze` command |
| `tests/test_analysis_loader.py` | Input loading and normalization tests |
| `tests/test_analysis_statistics.py` | Metrics, ranking, and completeness tests |
| `tests/test_analysis_outputs.py` | Chart, CSV, and Markdown output tests |

---

### Task 1: Result Loading and Normalization

**Files:**
- Create: `analysis/__init__.py`
- Create: `analysis/loader.py`
- Create: `tests/test_analysis_loader.py`

**Interfaces:**
- Produces `load_results(results_dir: Path) -> pd.DataFrame`.
- Produces `normalize_results(frame: pd.DataFrame) -> pd.DataFrame`.
- Output columns include `detector`, `architecture`, `fold`, `model_id`, and recognized metric columns.

- [ ] **Step 1: Write failing loader tests**

```python
from pathlib import Path
import pandas as pd
from analysis.loader import load_results, normalize_results


def test_normalizes_legacy_ml_column():
    frame = pd.DataFrame({"ml": ["YOLOV8"], "fold": ["fold_1"], "mAP": [0.5]})
    result = normalize_results(frame)
    assert result.loc[0, "detector"] == "YOLOV8"
    assert result.loc[0, "model_id"] == "YOLOV8"


def test_loads_summary_csv_before_json(tmp_path: Path):
    pd.DataFrame({"detector": ["yolo"], "architecture": ["yolov8s"], "fold": [1], "mAP": [0.5]}).to_csv(tmp_path / "summary.csv", index=False)
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
```

- [ ] **Step 2: Run tests and verify the expected import/function failures**

```bash
pytest tests/test_analysis_loader.py -q
```

Expected: FAIL because `analysis.loader` does not exist yet.

- [ ] **Step 3: Implement normalization and loading**

Implement these exact behaviors:

```python
METRICS = ["mAP", "mAP50", "mAP75", "MAE", "RMSE", "precision", "recall", "f1", "pearson_r"]


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "detector" not in result and "ml" in result:
        result["detector"] = result["ml"]
    if "architecture" not in result:
        result["architecture"] = result.get("detector", "unknown")
    if "fold" not in result:
        result["fold"] = "unknown"
    result["model_id"] = result["detector"].astype(str) + "/" + result["architecture"].astype(str)
    for metric in METRICS:
        if metric in result:
            result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result
```

`load_results` must prefer `results_dir / "summary.csv"`; otherwise recursively load `metrics.json`, infer detector/architecture/fold from relative path parts, and raise a clear `FileNotFoundError` when neither source exists.

- [ ] **Step 4: Run loader tests**

```bash
pytest tests/test_analysis_loader.py -q
```

- [ ] **Step 5: Commit**

```bash
```

---

### Task 2: Statistics, Rankings, and Completeness

**Files:**
- Create: `analysis/statistics.py`
- Create: `tests/test_analysis_statistics.py`

**Interfaces:**
- `descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame`.
- `model_ranking(frame: pd.DataFrame) -> pd.DataFrame`.
- `fold_completeness(frame: pd.DataFrame, expected_folds: Optional[int]) -> pd.DataFrame`.

- [ ] **Step 1: Write failing statistics tests**

```python
import pandas as pd
from analysis.statistics import descriptive_statistics, model_ranking


def sample_results():
    return pd.DataFrame({
        "model_id": ["a", "a", "b", "b"],
        "detector": ["x"] * 4,
        "architecture": ["a", "a", "b", "b"],
        "fold": [1, 2, 1, 2],
        "mAP": [0.4, 0.6, 0.7, 0.8],
        "MAE": [4.0, 3.0, 2.0, 1.0],
    })


def test_statistics_include_median_and_iqr():
    result = descriptive_statistics(sample_results())
    row = result[(result.model_id == "a") & (result.metric == "mAP")].iloc[0]
    assert row["mean"] == 0.5
    assert row["median"] == 0.5
    assert row["iqr"] == 0.1


def test_error_metric_is_ranked_lower_is_better():
    result = model_ranking(sample_results())
    mae = result[result.metric == "MAE"].sort_values("rank")
    assert mae.iloc[0].model_id == "b"
```

- [ ] **Step 2: Run tests and verify failure**

```bash
pytest tests/test_analysis_statistics.py -q
```

- [ ] **Step 3: Implement statistics**

Convert each metric into long form, calculate `count`, `mean`, `median`, `std`, `min`, `q1`, `q3`, `max`, and `iqr`. Rank group means by metric direction. Calculate an overall score by min-max normalizing each metric across models, inverting `MAE` and `RMSE`, then averaging available normalized metrics.

Completeness must report `observed_folds`, `expected_folds`, `missing_folds`, and `complete` for each `model_id`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_analysis_statistics.py -q
```

- [ ] **Step 5: Commit**

```bash
```

---

### Task 3: Charts and Markdown Report

**Files:**
- Create: `analysis/plots.py`
- Create: `analysis/report.py`
- Create: `tests/test_analysis_outputs.py`

**Interfaces:**
- `create_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]`.
- `write_report(frame, stats, rankings, completeness, chart_paths, output_path: Path) -> None`.

- [ ] **Step 1: Write failing output tests**

```python
from pathlib import Path
import pandas as pd
from analysis.plots import create_plots
from analysis.report import write_report


def test_outputs_charts_and_report(tmp_path: Path):
    frame = pd.DataFrame({
        "model_id": ["a", "a", "b", "b"],
        "fold": [1, 2, 1, 2],
        "mAP": [0.4, 0.6, 0.7, 0.8],
        "MAE": [4.0, 3.0, 2.0, 1.0],
    })
    charts = create_plots(frame, tmp_path / "charts")
    assert any(path.name == "boxplot_map.png" for path in charts)

    report = tmp_path / "findings.md"
    write_report(frame, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), charts, report)
    assert "# Detector Results Analysis" in report.read_text()
```

- [ ] **Step 2: Run tests and verify failure**

```bash
pytest tests/test_analysis_outputs.py -q
```

- [ ] **Step 3: Implement plotting**

Set `matplotlib.use("Agg")`. Generate one boxplot per metric group, with model IDs on the x-axis and fold observations as values. Generate `metric_means.png` from grouped means. Skip charts for metrics with no usable data and return only paths that were created.

- [ ] **Step 4: Implement Markdown reporting**

Write an English report with sections for dataset coverage, best models per metric, overall ranking, stability, warnings, and chart links. Explicitly state metric direction and avoid significance claims.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_analysis_outputs.py -q
```

- [ ] **Step 6: Commit**

```bash
```

---

### Task 4: Analysis Pipeline and CLI Command

**Files:**
- Create: `analysis/pipeline.py`
- Modify: `cli.py`

**Interfaces:**
- `run_analysis(results_dir: Path, output_dir: Path, expected_folds: Optional[int] = None) -> dict[str, Path]`.
- CLI: `python cli.py analyze --results <path> --output <path> [--expected-folds N]`.

- [ ] **Step 1: Write failing CLI test**

```python
from typer.testing import CliRunner
from cli import app


def test_analyze_command_is_registered():
    result = CliRunner().invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--results" in result.stdout
```

- [ ] **Step 2: Run test and verify failure**

```bash
pytest tests/test_analysis_outputs.py::test_analyze_command_is_registered -q
```

- [ ] **Step 3: Implement pipeline and CLI command**

The pipeline will load results, write `normalized_results.csv`, `descriptive_statistics.csv`, `model_ranking.csv`, and `fold_completeness.csv`, generate charts, and write `findings.md`. The CLI command must create the requested output directory and report its output paths.

- [ ] **Step 4: Run the full analysis tests**

```bash
pytest tests/test_analysis_loader.py tests/test_analysis_statistics.py tests/test_analysis_outputs.py -q
```

- [ ] **Step 5: Run a real analysis against a result directory**

```bash
python cli.py analyze --results results/yolov8_faster --output /tmp/yolov8_faster_analysis
```

Expected: CSV files, PNG charts, and `findings.md` are created without modifying `results/yolov8_faster`.

- [ ] **Step 6: Commit**

```bash
```

---

### Task 5: Final Verification and Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the analyze command**

Add the command, expected result directory structure, generated files, and interpretation notes to the README.

- [ ] **Step 2: Run all available tests**

```bash
pytest -q
```

- [ ] **Step 3: Verify CLI help and clean diff**

```bash
python cli.py analyze --help
```

- [ ] **Step 4: Commit documentation**

```bash
```
