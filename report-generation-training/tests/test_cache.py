import json

import numpy as np
import pytest
import torch

from mrrate_report_training.cache import ExactRaggedTokenDataset
from mrrate_report_training.targets import make_report_target


def make_cache(root, max_tokens=0):
    tokens = np.arange(30, dtype=np.float16).reshape(6, 5)
    token_file = root / "tokens_train_test.bin"
    tokens.tofile(token_file)
    np.save(root / "offsets.npy", np.array([0, 2, 6], dtype=np.int64))
    np.save(root / "labels.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
    np.save(root / "full.npy", np.array([2, 4], dtype=np.int64))
    np.save(root / "series.npy", np.array([1, 2], dtype=np.int32))
    (root / "ids.txt").write_text("a\nb\n")
    (root / "label_names.json").write_text(json.dumps(["x", "y"]))
    manifest = {
        "format": "raw_numpy_memmap",
        "feature_level": "projected_per_series_visual_tokens",
        "max_tokens_per_study": max_tokens,
        "dim": 5,
        "dtype": "float16",
        "tokens_file": token_file.name,
        "offsets_file": "offsets.npy",
        "labels_file": "labels.npy",
        "subject_ids_file": "ids.txt",
        "full_token_counts_file": "full.npy",
        "series_counts_file": "series.npy",
    }
    (root / "token_features_train.json").write_text(json.dumps(manifest))
    return torch.from_numpy(tokens.copy())


def test_exact_cache_preserves_online_tokens(tmp_path):
    online = make_cache(tmp_path)
    targets = {
        value: make_report_target(value, "There is no abnormality.")
        for value in ("a", "b")
    }
    cached = ExactRaggedTokenDataset(
        tmp_path,
        "train",
        targets,
        expected_dim=5,
        expected_label_names=["x", "y"],
    )
    assert torch.equal(cached[0]["tokens"], online[:2])
    assert torch.equal(cached[1]["tokens"], online[2:])


def test_capped_cache_is_rejected(tmp_path):
    make_cache(tmp_path, max_tokens=100)
    targets = {
        value: make_report_target(value, "There is no abnormality.")
        for value in ("a", "b")
    }
    with pytest.raises(ValueError, match="max_tokens_per_study=0"):
        ExactRaggedTokenDataset(tmp_path, "train", targets, expected_dim=5)
