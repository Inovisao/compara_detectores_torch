from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from Detectors.FasterRCNN.geradataset import geredata


def test_filesjson_layout_uses_split_specific_image_dirs(tmp_path):
    files_json = tmp_path / "filesJSON"
    files_json.mkdir()
    (files_json / "fold_1_train.json").write_text("{}", encoding="utf-8")
    (files_json / "fold_1_val.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fold_1" / "train" / "images").mkdir(parents=True)
    (tmp_path / "fold_1" / "val" / "images").mkdir(parents=True)

    cfg = geredata("fold_1", tmp_path)

    assert cfg.train_dir == tmp_path.resolve() / "fold_1" / "train" / "images"
    assert cfg.val_dir == tmp_path.resolve() / "fold_1" / "val" / "images"
