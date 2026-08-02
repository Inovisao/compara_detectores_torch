# Task 2 Report: Statistics, Rankings, and Completeness

## Status

Implemented Task 2 only. Added descriptive statistics, direction-aware model
rankings, normalized overall scores, and fold completeness reporting using the
normalized `model_id`, `fold`, and recognized metric columns from Task 1.

Missing metric values are coerced to numeric where needed and excluded from
metric aggregates and normalized-score averages. `MAE` and `RMSE` are ranked
lower-is-better; all other recognized metrics are ranked higher-is-better.
The implementation uses typing and syntax compatible with Python 3.9.

## Files

- `analysis/statistics.py`
- `tests/test_analysis_statistics.py`

## Interfaces

- `descriptive_statistics(frame)` returns one long-form row per model and
  metric with `count`, `mean`, `median`, `std`, `min`, `q1`, `q3`, `max`, and
  `iqr`.
- `model_ranking(frame)` returns per-metric means, direction-aware ranks,
  min-max normalized scores, and `overall` rows containing the mean available
  normalized score and overall rank.
- `fold_completeness(frame, expected_folds)` reports observed fold counts,
  expected fold counts, missing fold identifiers, and completion status for
  each model. With no explicit expectation, the observed global fold set is
  used as the expectation.

## Commit

Implementation commit: `5040c95` (`feat: add results statistics and rankings`)

Report commit: the commit containing this report.

## TDD Evidence

### Required red run

After adding the tests and before implementing the module:

```text
$ pytest tests/test_analysis_statistics.py -q
ERROR collecting tests/test_analysis_statistics.py
ModuleNotFoundError: No module named 'analysis.statistics'
```

The failure was caused by the intended missing production module.

### Focused green run

```text
$ pytest tests/test_analysis_statistics.py -q
......                                                                   [100%]
6 passed in 0.18s
```

### Loader regression run

```text
$ pytest tests/test_analysis_loader.py tests/test_analysis_statistics.py -q
..................                                                       [100%]
18 passed in 0.20s
```

### Python 3.9 compatibility check

```text
$ /home/maxfukui/miniconda3/envs/compara_detectores/bin/python -m py_compile analysis/statistics.py tests/test_analysis_statistics.py
```

Completed with exit code 0 and no output.

### Diff validation

```text
$ git diff --check
```

Completed with exit code 0 and no output before the implementation commit.

### Full test suite

```text
$ pytest -q
3 errors during collection
```

The existing unrelated tests cannot collect because the active environment is
missing `cv2` and `sklearn`.

## Concerns

- Full-suite verification remains blocked by pre-existing environment omissions
  (`cv2` and `sklearn`); the focused Task 1 and Task 2 suites pass.
- Constant metrics receive a normalized score of `1.0` for every model because
  min-max normalization has no distinguishing range.
- Explicit `expected_folds=N` represents the conventional fold identifiers
  `1..N`; labels such as `fold_1` are normalized to the corresponding integer.

## Review Fix Report

### Status

Fixed both Task 2 review findings.

- `_metric_frame` now preserves every model/metric combination after numeric
  coercion. Descriptive statistics emits an all-missing row with count `0`
  and NaN statistics instead of dropping the combination. Ranking continues to
  exclude missing observations from score calculations.
- `_fold_key` now normalizes numeric strings such as `"1"` to integer fold
  identifiers, in addition to the existing `fold_1` form.

### Added Regression Tests

- `test_statistics_preserve_all_missing_model_metric_combination`
- `test_fold_completeness_normalizes_numeric_string_folds`

### Verification

```text
$ pytest tests/test_analysis_statistics.py -q
........                                                                 [100%]
8 passed in 0.18s
```

```text
$ pytest tests/test_analysis_loader.py tests/test_analysis_statistics.py -q
....................                                                     [100%]
20 passed in 0.20s
```

```text
$ /home/maxfukui/miniconda3/envs/compara_detectores/bin/python -m py_compile analysis/statistics.py tests/test_analysis_statistics.py
```

Completed with exit code 0 and no output.

```text
$ git diff --check
```

Completed with exit code 0 and no output before the fix commit.

### Concerns

- The full repository suite remains blocked by the pre-existing environment
  omissions of `cv2` and `sklearn`.
