from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import ReportTarget


class ExactRaggedTokenDataset(Dataset):
    """Exact projected visual-token bags aligned with complete report targets."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        targets: dict[str, ReportTarget],
        *,
        expected_dim: int = 512,
        expected_label_names: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        manifest_path = self.cache_dir / f"token_features_{split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing exact token manifest: {manifest_path}")
        self.metadata = json.loads(manifest_path.read_text())
        if self.metadata.get("format") != "raw_numpy_memmap":
            raise ValueError("Cached training needs the raw_numpy_memmap token cache")
        if self.metadata.get("feature_level") != "projected_per_series_visual_tokens":
            raise ValueError("Pooled features cannot be used for report attention")
        if int(self.metadata.get("max_tokens_per_study", -1)) != 0:
            raise ValueError("Exact training requires max_tokens_per_study=0")
        self.dim = int(self.metadata["dim"])
        if self.dim != int(expected_dim):
            raise ValueError(f"Token dim {self.dim} != expected {expected_dim}")
        self.dtype = np.dtype(self.metadata["dtype"])
        if self.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError(f"Unsupported cache dtype: {self.dtype}")

        def load_array(key: str) -> np.ndarray:
            return np.load(self.cache_dir / self.metadata[key])

        self.offsets = load_array("offsets_file").astype(np.int64, copy=False)
        self.labels = load_array("labels_file").astype(np.float32, copy=False)
        ids_path = self.cache_dir / self.metadata["subject_ids_file"]
        self.subject_ids = ids_path.read_text().strip().splitlines()
        self.full_counts = load_array("full_token_counts_file").astype(
            np.int64, copy=False
        )
        self.series_counts = load_array("series_counts_file").astype(
            np.int64, copy=False
        )
        if len(self.subject_ids) != len(set(self.subject_ids)):
            raise ValueError(f"Duplicate subject IDs in {ids_path}")
        if self.offsets.shape != (len(self.subject_ids) + 1,):
            raise ValueError("Offsets and subject IDs are misaligned")
        if self.labels.shape[0] != len(self.subject_ids):
            raise ValueError("Labels and subject IDs are misaligned")
        if self.full_counts.shape != (len(self.subject_ids),):
            raise ValueError("Full token counts are misaligned")
        if self.series_counts.shape != (len(self.subject_ids),):
            raise ValueError("Series counts are misaligned")
        cached_counts = np.diff(self.offsets)
        if np.any(cached_counts <= 0) or not np.array_equal(
            cached_counts, self.full_counts
        ):
            raise ValueError("Cache is empty, capped, or has inconsistent token counts")
        missing = [subject_id for subject_id in self.subject_ids if subject_id not in targets]
        if missing:
            raise ValueError(
                f"{len(missing)} cached studies lack report targets; first={missing[:5]}"
            )
        self.targets = targets
        if expected_label_names is not None:
            names_path = self.cache_dir / "label_names.json"
            if not names_path.exists():
                raise FileNotFoundError(f"Missing label schema: {names_path}")
            names = json.loads(names_path.read_text())
            if names != list(expected_label_names):
                raise ValueError("Token cache label schema differs from MIL checkpoint")
        self.num_tokens = int(self.offsets[-1])
        token_path = self.cache_dir / self.metadata["tokens_file"]
        expected_bytes = self.num_tokens * self.dim * self.dtype.itemsize
        if token_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Token memmap has {token_path.stat().st_size} bytes, "
                f"expected {expected_bytes}"
            )
        self.tokens = np.memmap(
            token_path,
            mode="r",
            dtype=self.dtype,
            shape=(self.num_tokens, self.dim),
        )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        subject_id = self.subject_ids[index]
        return {
            "subject_id": subject_id,
            "tokens": torch.from_numpy(np.array(self.tokens[start:end], copy=True)),
            "mil_labels": torch.from_numpy(np.array(self.labels[index], copy=True)),
            "target": self.targets[subject_id],
            "series_count": int(self.series_counts[index]),
        }


def collate_single_study(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(
            "MR token bags are intentionally trained one study/GPU at a time"
        )
    return batch[0]
