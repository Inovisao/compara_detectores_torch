# Results Prototype

This directory is reserved for prototype detector results and their analysis.

Expected inputs:

- `summary.csv` at this directory root, or
- Nested `metrics.json` files under fold/model directories.

Run the analysis after result files are available:

```bash
python cli.py analyze \
  --results results_prototype \
  --output results_prototype_analysis
```

No result CSV or JSON files are currently present, so no findings are generated yet.
