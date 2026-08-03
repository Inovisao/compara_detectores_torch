from pathlib import Path


def test_pipeline_script_runs_training_then_analysis():
    script = Path("run_training_and_analysis.sh").read_text()
    assert "cli.py sweep" in script
    assert "cli.py analyze" in script
    assert script.index("cli.py sweep") < script.index("cli.py analyze")


def test_pipeline_script_supports_analysis_only_mode():
    script = Path("run_training_and_analysis.sh").read_text()
    assert "--analysis-only" in script
