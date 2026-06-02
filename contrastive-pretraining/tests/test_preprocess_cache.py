"""Tests for the offline .npz preprocessing cache.

Covers:
  - discover_subjects() finds subjects independent of reports/splits
  - preprocess_volumes.py writes a manifest + one .npz per subject
  - the cached path is numerically identical to the live preprocessing path
  - the cache manifest guards against config mismatch
"""
import json
import os
import runpy
import sys

import numpy as np
import pytest
import torch

from data import (
    MRReportDataset, discover_subjects, preprocess_nii, NORMALIZERS,
    build_cache_manifest, validate_cache_manifest, CACHE_MANIFEST_NAME,
)

# synthetic_dataset fixture lives in test_data.py; import it so this file can use it.
from test_data import synthetic_dataset  # noqa: F401


def _run_preprocess(argv):
    """Invoke preprocess_volumes.py main() with the given argv (no subprocess)."""
    old = sys.argv
    sys.argv = ["preprocess_volumes.py"] + argv
    try:
        runpy.run_module("preprocess_volumes", run_name="__main__")
    finally:
        sys.argv = old


class TestDiscoverSubjects:
    def test_finds_all_regardless_of_reports(self, synthetic_dataset):
        """discover_subjects ignores reports/splits — all 4 subjects show up."""
        found = discover_subjects(synthetic_dataset["mri_dir"], "native_space")
        ids = {s["subject_id"] for s in found}
        assert ids == {"SUBJ_AAA", "SUBJ_BBB", "SUBJ_CCC", "SUBJ_DDD"}
        # volume counts preserved
        by_id = {s["subject_id"]: len(s["image_paths"]) for s in found}
        assert by_id["SUBJ_AAA"] == 2 and by_id["SUBJ_BBB"] == 3


class TestPreprocessScript:
    def test_writes_manifest_and_npz(self, synthetic_dataset, tmp_path):
        out_dir = tmp_path / "cache"
        _run_preprocess([
            "--data_folder", synthetic_dataset["mri_dir"],
            "--out_dir", str(out_dir),
            "--space", "native_space",
            "--target_shape", "8", "8", "8",
            "--num_workers", "1",
        ])
        space_dir = out_dir / "native_space"
        # manifest present and well-formed
        manifest = json.loads((space_dir / CACHE_MANIFEST_NAME).read_text())
        assert manifest["space"] == "native_space"
        assert manifest["target_shape"] == [8, 8, 8]
        assert manifest["normalizer"] == "zscore"
        # one npz per subject, each [N, 8, 8, 8]
        npzs = sorted(p for p in os.listdir(space_dir) if p.endswith(".npz"))
        assert npzs == ["SUBJ_AAA.npz", "SUBJ_BBB.npz", "SUBJ_CCC.npz", "SUBJ_DDD.npz"]
        aaa = np.load(space_dir / "SUBJ_AAA.npz")["volumes"]
        assert aaa.shape == (2, 8, 8, 8)
        assert aaa.dtype == np.float16

    def test_resume_skips_existing(self, synthetic_dataset, tmp_path):
        out_dir = tmp_path / "cache"
        args = [
            "--data_folder", synthetic_dataset["mri_dir"],
            "--out_dir", str(out_dir), "--space", "native_space",
            "--target_shape", "8", "8", "8", "--num_workers", "1",
        ]
        _run_preprocess(args)
        npz = out_dir / "native_space" / "SUBJ_AAA.npz"
        mtime = npz.stat().st_mtime_ns
        # Second run without --overwrite must not rewrite the file
        _run_preprocess(args)
        assert npz.stat().st_mtime_ns == mtime


class TestCacheEquivalence:
    def test_cached_matches_live(self, synthetic_dataset, tmp_path):
        """Cache path must yield byte-for-byte the same bf16 tensor as live path."""
        out_dir = tmp_path / "cache"
        _run_preprocess([
            "--data_folder", synthetic_dataset["mri_dir"],
            "--out_dir", str(out_dir), "--space", "native_space",
            "--target_shape", "8", "8", "8", "--num_workers", "1",
            "--dtype", "float32",  # match live float32->bf16 exactly (no fp16 rounding)
        ])

        common = dict(
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
            target_shape=(8, 8, 8),
        )
        live = MRReportDataset(data_folder=synthetic_dataset["mri_dir"], **common)
        cached = MRReportDataset(
            data_folder=None, use_preprocessed=True,
            preprocessed_dir=str(out_dir), **common,
        )

        live_by_id = {s["subject_id"]: i for i, s in enumerate(live.samples)}
        cached_by_id = {s["subject_id"]: i for i, s in enumerate(cached.samples)}
        assert set(live_by_id) == set(cached_by_id)

        for sid in live_by_id:
            lv = live[live_by_id[sid]][0]   # [N,1,D,H,W] bf16
            cv = cached[cached_by_id[sid]][0]
            assert lv.shape == cv.shape
            assert lv.dtype == torch.bfloat16 and cv.dtype == torch.bfloat16
            assert torch.equal(lv, cv), f"mismatch for {sid}"


class TestManifestValidation:
    def _make_manifest(self, space_dir):
        os.makedirs(space_dir, exist_ok=True)
        man = build_cache_manifest(
            "native_space", (1.0, 0.5, 0.5), (256, 384, 384), 15.0,
            "zscore", {}, "float16",
        )
        with open(os.path.join(space_dir, CACHE_MANIFEST_NAME), "w") as f:
            json.dump(man, f)

    def test_matching_config_passes(self, tmp_path):
        space_dir = tmp_path / "native_space"
        self._make_manifest(space_dir)
        # Should not raise
        validate_cache_manifest(
            str(space_dir), "native_space", (1.0, 0.5, 0.5), (256, 384, 384),
            15.0, "zscore", {},
        )

    def test_normalizer_mismatch_raises(self, tmp_path):
        space_dir = tmp_path / "native_space"
        self._make_manifest(space_dir)
        with pytest.raises(ValueError, match="does not match"):
            validate_cache_manifest(
                str(space_dir), "native_space", (1.0, 0.5, 0.5), (256, 384, 384),
                15.0, "percentile", {},
            )

    def test_mismatch_allowed_with_flag(self, tmp_path, capsys):
        space_dir = tmp_path / "native_space"
        self._make_manifest(space_dir)
        validate_cache_manifest(
            str(space_dir), "native_space", (1.0, 0.5, 0.5), (256, 384, 384),
            15.0, "percentile", {}, allow_mismatch=True,
        )
        assert "WARNING" in capsys.readouterr().out

    def test_missing_manifest_raises(self, tmp_path):
        space_dir = tmp_path / "native_space"
        os.makedirs(space_dir)
        with pytest.raises(FileNotFoundError):
            validate_cache_manifest(
                str(space_dir), "native_space", (1.0, 0.5, 0.5), (256, 384, 384),
                15.0, "zscore", {},
            )
