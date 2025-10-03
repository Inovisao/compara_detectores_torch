### General Instructions 
- You are a Product Manager and you will help me to set an experiment to add into a server for future training.

### Quick Background
- The problem is that the images are big so we are using tiling and grid methods to make it possible to train in one single GPU.
- The folder (root) is about a script for comparing detection models. 
- The folder ./create_dataset/ is related to creation of tiling and grid for the images. 
- So there are 2 repositories (scripts) that will help us with that. 
- After changes and modifications, please, git commit. 
- As this repository is not mine, please create a new branch for this and just commit in this new branch. 


### History

2025-10-03: Initial assessment performed. Reviewed create_dataset tiling engine and dataset structure; proposed plan added below.


### TO-DO: What I said for the eng_co.md file (developer to-do)
Comment: I reviewed the repository structure and the create_dataset toolkit to propose a concrete plan that satisfies your constraints (grid tiling, k-folds, train/val filtering, full test coverage) while minimizing code changes.

Strategy overview (current state):
- There is a dedicated tiling app in ./create_dataset with grid mode support (create_dataset/src/core/tiling/engine.py) and a processor (create_dataset/src/services/dataset/processor.py) that writes COCO annotations.
- The processor currently expects a single COCO file at <input>/train/_annotations.coco.json and always saves all generated tiles (even tiles without annotations) under <output>/train/.
- In this repo, the folds are provided as COCO JSONs in ./dataset/all/filesJSON/ (e.g., fold_1_train.json, fold_1_val.json, fold_1_test.json). The images appear to be under ./dataset/all/train/ in this repository snapshot. Please confirm if images should be read from ./dataset/all/train/ or ./dataset/all/images/.

Comprehensive developer to-do list (scoped, minimal changes):
1) Fold-aware tiling entrypoint
- Add a small wrapper script (preferred location: repo root scripts/ or utils/) to drive the create_dataset tiler across folds and splits.
  - For each k in {1..6} and each split in {train, val, test}:
    - Input JSON: ./dataset/all/filesJSON/fold_{k}_{split}.json
    - Source images dir: to be confirmed (see Questions). Detect automatically by joining the file_name entries to the candidate dirs and checking existence.
    - Output root: ./dataset/tiles/grid/ (or another agreed prefix), creating: ./dataset/tiles/grid/fold_{k}/{split}/images and {split}/_annotations.coco.json
  - Pass grid parameters to the tiler via config/CLI.

2) Train/val filtering vs. test keep-all
- Implement an optional flag in create_dataset/src/services/dataset/processor.py (config-driven) to skip saving tiles that end up with zero annotations.
  - Proposed config flag: keep_empty_tiles: bool (default False)
  - Behavior:
    - When False: do not save tile image nor add image entry if tile_annotations is empty (this satisfies train and val requirements).
    - When True: save all tiles and include empty ones (this satisfies test requirement).
  - The change is small and localized where tiles are saved and annotations are appended.

3) COCO ingestion per split
- Extend the processor to read an arbitrary annotations_path (not only <input>/train/_annotations.coco.json). Expose it via CLI argument --annotations and pass the fold_x_split.json for each run.
- Ensure output is written to <output>/<split>/_annotations.coco.json with images under <output>/<split>/.

4) K-folds orchestration
- Write a driver that loops 6 folds and 3 splits, invoking the processor with:
  - keep_empty_tiles=False for train and val
  - keep_empty_tiles=True for test
  - consistent tiling parameters (grid rows/cols or tile size/overlap) as provided by you
- Optional: parallelize per fold if I/O allows.

5) New cleaner utility (only if we choose not to filter at tiling time)
- If you prefer not to modify the processor, create utils/filter_coco_by_nonempty_images.py:
  - Reads a COCO JSON and removes image entries without annotations and deletes corresponding image files from disk.
  - This will be applied only to train and val after tiling. However, modifying the processor is cleaner and avoids wasted I/O.

6) Determinism and metadata
- Add run metadata to each output folder (params.json) capturing: grid/tile params, keep_empty_tiles, min_object_coverage, source fold JSON path, timestamp, git commit.
- Maintain original category ids and image file stems in new filenames for traceability.

7) Sanity checks and reporting
- After each split is generated, run a validator (processor.validate_output()) and a small summary report: number of images, annotations, per-category counts. Save to report.txt in each split dir.

8) Training integration
- Confirm your training script expects COCO folders per split in the usual structure. If it expects a consolidated data.yaml, generate one pointing to the new tiled fold paths.

9) Documentation
- Add a short README section with exact commands to produce all 6 folds and expected output structure.

Questions for you (to finalize parameters and paths):
- Source images dir: should we read from ./dataset/all/train/ (as in this repo snapshot) or ./dataset/all/images/ (as stated in your note)?
- Tiling parameters: grid_rows x grid_cols OR tile_size with overlap? Desired values? Any resizing of tiles (e.g., 1024x1024)?
- Minimum object coverage threshold (min_object_coverage) to keep partial boxes in a tile (current default in tiler is used). What threshold do you want?
- For train/val, do you want to keep tiles with extremely small coverage of an object (e.g., <5%) if they still have a COCO box, or strictly zero-annotation tiles only should be discarded? The processor can enforce either policy.
- Output directory naming preference: dataset/tiles/grid/fold_{k}/{split}/ or another convention?
- Any class filtering or remapping required, or keep categories as-is?

Acceptance criteria
- For each fold k in {1..6}, directories exist for train, val, test with images/ and _annotations.coco.json.
- Train/val contain only tiles that have at least one annotation entry; test contains all tiles.
- Summary reports exist and counts look reasonable; validator passes.

```
Developer checklist
- [ ] Add keep_empty_tiles config + CLI to processor and conditionally skip empty tiles.
- [ ] Add CLI argument --annotations to processor to allow arbitrary annotations path and split-aware output dirs.
- [ ] Implement a fold driver script to iterate 6 folds x 3 splits with proper flags and paths.
- [ ] Generate outputs under agreed root and write metadata + summary reports.
- [ ] Document the commands to reproduce.
- [ ] (If not modifying processor) Implement utils/filter_coco_by_nonempty_images.py and wire it post-tiling for train/val.
```
