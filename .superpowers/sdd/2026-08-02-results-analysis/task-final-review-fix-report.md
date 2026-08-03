# Results Analysis Final Review Fixes

**Date:** 2026-08-02

## Fixes

- Unusable non-empty `summary.csv` files now fall back to nested `metrics.json` files.
- JSON fallback metadata recognizes `fold_N/<architecture>/<combo_name>/metrics.json`, preserving detector, architecture, and combo identity in `model_id`.
- Normalized outputs flag duplicate model/fold rows and report duplicate groups. Aggregate statistics retain the first observation for a duplicate key and exclude repeated rows from calculations.
- Boxplot and mean-chart labels rotate when many model combinations are present.
- The approved spec example now places analysis output outside the raw results directory and documents the collision-protection rule.

## Verification

- Focused analysis tests: `34 passed`.
- Compilation: `python -m compileall -q analysis cli.py` passed.
- Diff validation: `git diff --check` passed.
- Full suite: blocked during collection because the environment lacks `cv2` and `sklearn`; three unrelated test modules could not import.
