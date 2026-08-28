"""Coregistered multi-sequence MR sampling for native 3-D DINOv3.

The dataset consumes the exact cache produced by MR-RATE's existing
``contrastive-pretraining/scripts/preprocess_volumes.py``.  A cache item is one
study with aligned sequences in ``volumes[N, D, H, W]``.  Global and local DINO
views share a physical region, but may use different aligned MR sequences.
Every iBOT teacher/student pair always uses the same sequence and voxel crop.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


CACHE_MANIFEST_NAME = "_manifest.json"
EXPECTED_CACHE_LAYOUT = "per_subject_stack"


@dataclass(frozen=True)
class CropSpec:
    global_shape: tuple[int, int, int] = (64, 192, 192)
    local_shape: tuple[int, int, int] = (32, 96, 96)
    local_crops: int = 8


class InfiniteStudySampler(Sampler[int]):
    """Grouped sequence stream with exact resume and one-pass coverage.

    Studies are shuffled each epoch, then their sequence anchors are shuffled
    within-study. Keeping a study's anchors adjacent lets each DataLoader worker
    reuse the large per-study NPZ stack instead of decoding it once per anchor.
    """

    def __init__(
        self,
        n: int,
        seed: int,
        offset: int = 0,
        group_sizes: list[int] | None = None,
    ) -> None:
        if n < 1:
            raise ValueError("Each rank must own at least one MR study")
        self.n = int(n)
        self.seed = int(seed)
        self.offset = int(offset)
        self.group_sizes = [int(x) for x in group_sizes] if group_sizes else [1] * self.n
        if any(x < 1 for x in self.group_sizes) or sum(self.group_sizes) != self.n:
            raise ValueError("group_sizes must be positive and sum to n")
        self.group_offsets = np.cumsum([0] + self.group_sizes[:-1]).tolist()

    def set_offset(self, offset: int) -> None:
        self.offset = int(offset)

    def __iter__(self):
        while True:
            epoch, pos = divmod(self.offset, self.n)
            generator = torch.Generator().manual_seed(self.seed + epoch)
            study_order = torch.randperm(len(self.group_sizes), generator=generator).tolist()
            order = []
            for study in study_order:
                within = torch.randperm(self.group_sizes[study], generator=generator).tolist()
                order.extend(self.group_offsets[study] + index for index in within)
            for j in range(pos, self.n):
                self.offset += 1
                yield order[j], epoch
            pos = 0

    def __len__(self) -> int:
        return 2**31


def _random_start(
    container_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    return tuple(
        rng.randrange(have - want + 1) if have > want else 0
        for have, want in zip(container_shape, crop_shape)
    )


def _crop_at_start(
    x: torch.Tensor,
    shape: tuple[int, int, int],
    start: tuple[int, int, int],
) -> torch.Tensor:
    """Crop ``[D,H,W]`` at an explicit start, padding outside with zero."""
    out = x.new_zeros(shape)
    spans = [max(0, min(have - begin, want)) for have, want, begin in zip(x.shape, shape, start)]
    z, y, xx = start
    d, h, w = spans
    if d and h and w:
        out[:d, :h, :w] = x[z : z + d, y : y + h, xx : xx + w]
    return out


def _overlapping_global_starts(
    container_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    rng: random.Random,
    first: tuple[int, int, int] | None = None,
    min_overlap: float = 0.75,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return two starts with at least ``min_overlap`` on every axis."""
    first = first or _random_start(container_shape, crop_shape, rng)
    second = []
    for have, want, origin in zip(container_shape, crop_shape, first):
        if have <= want:
            second.append(0)
            continue
        max_shift = max(0, int((1.0 - min_overlap) * want))
        lo = max(0, origin - max_shift)
        hi = min(have - want, origin + max_shift)
        second.append(rng.randint(lo, hi))
    return first, tuple(second)


def _intersection_box(
    starts: tuple[tuple[int, int, int], tuple[int, int, int]],
    shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    lo = tuple(max(a, b) for a, b in zip(*starts))
    hi = tuple(min(a + n, b + n) for a, b, n in zip(starts[0], starts[1], shape))
    return lo, hi


def _contained_start(
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    starts = []
    for left, right, want in zip(lo, hi, crop_shape):
        if right - left < want:
            raise ValueError(f"Local crop {crop_shape} does not fit intersection {(lo, hi)}")
        starts.append(rng.randrange(left, right - want + 1) if right - left > want else left)
    return tuple(starts)


def _informative_start(
    reference: torch.Tensor,
    crop_shape: tuple[int, int, int],
    rng: random.Random,
    trials: int,
) -> tuple[int, int, int]:
    """Choose randomly among informative candidates, avoiding all-padding crops."""
    candidates = []
    for _ in range(max(1, trials)):
        start = _random_start(tuple(reference.shape), crop_shape, rng)
        crop = _crop_at_start(reference, crop_shape, start).float()
        score = float(crop.std()) + 0.05 * float(crop.abs().mean())
        candidates.append((score, start))
    candidates.sort(key=lambda item: item[0], reverse=True)
    pool = candidates[: max(1, len(candidates) // 2)]
    return rng.choice(pool)[1]


def _student_distortion(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """MR-safe intensity augmentation without anatomical-axis reversal."""
    y = x.float()
    if rng.random() < 0.8:
        y = y * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    if rng.random() < 0.5:
        generator = torch.Generator().manual_seed(rng.randrange(2**63 - 1))
        y = y + torch.randn(y.shape, generator=generator) * rng.uniform(0.005, 0.04)
    if rng.random() < 0.25:
        y = F.avg_pool3d(y[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    return y.clamp_(-1, 1).to(torch.bfloat16)


def _load_split_ids(splits_csv: str | None, split: str) -> set[str] | None:
    if not splits_csv:
        return None
    selected: set[str] = set()
    with open(splits_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        id_column = "study_uid" if "study_uid" in fields else "subject_id"
        if id_column not in fields or "split" not in fields:
            raise ValueError(
                f"{splits_csv} must contain split and study_uid/subject_id columns; got {fields}"
            )
        for row in reader:
            if row["split"].strip() == split:
                selected.add(row[id_column].strip())
    return selected


def _cache_space_dir(preprocessed_dir: str, space: str) -> Path:
    root = Path(preprocessed_dir)
    nested = root / space
    return nested if nested.is_dir() else root


def validate_coreg_cache(
    preprocessed_dir: str,
    space: str = "coreg_space",
    expected_spacing: tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> dict:
    """Fail early unless this is a compatible MR-RATE coregistered cache."""
    if space != "coreg_space":
        raise ValueError("MR DINO requires space='coreg_space'; native/atlas inputs are not accepted")
    space_dir = _cache_space_dir(preprocessed_dir, space)
    manifest_path = space_dir / CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing MR-RATE preprocessing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("space") != "coreg_space":
        raise ValueError(f"Cache is {manifest.get('space')!r}, expected 'coreg_space'")
    if manifest.get("layout") != EXPECTED_CACHE_LAYOUT:
        raise ValueError(
            f"Unsupported cache layout {manifest.get('layout')!r}; expected {EXPECTED_CACHE_LAYOUT!r}"
        )
    spacing = tuple(float(x) for x in manifest.get("target_spacing", ()))
    if spacing != tuple(expected_spacing):
        raise ValueError(f"Cache spacing {spacing} does not match requested {expected_spacing}")
    return manifest


def discover_coreg_cache(
    preprocessed_dir: str,
    splits_csv: str | None = None,
    split: str = "train",
    space: str = "coreg_space",
) -> list[dict]:
    manifest = validate_coreg_cache(preprocessed_dir, space)
    space_dir = _cache_space_dir(preprocessed_dir, space)
    selected = _load_split_ids(splits_csv, split)
    samples = []
    for path in sorted(space_dir.glob("*.npz")):
        study_uid = path.stem
        if selected is not None and study_uid not in selected:
            continue
        shape = npz_volume_shape(path)
        samples.append({
            "study_uid": study_uid,
            "cache_path": str(path),
            "n_sequences": shape[0],
            "volume_shape": shape[1:],
        })
    if not samples:
        raise RuntimeError(f"No {split!r} coregistered studies found under {space_dir}")
    return samples


def npz_volume_shape(path: str | Path) -> tuple[int, int, int, int]:
    """Read ``volumes.npy`` shape from an NPZ header without loading its voxels."""
    with zipfile.ZipFile(path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        if "volumes.npy" not in members:
            raise ValueError(f"{path} has no volumes.npy member")
        with archive.open(members["volumes.npy"]) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}")
    shape = tuple(int(x) for x in shape)
    if len(shape) != 4 or shape[0] < 1:
        raise ValueError(f"Expected volumes[N,D,H,W] in {path}, got {shape}")
    return shape


def _discover_raw_coreg(data_folder: str, selected: set[str] | None) -> list[dict]:
    """Discover either MR-RATE batch layout or study/space/img layout."""
    root = Path(data_folder)
    samples = []
    direct = sorted(root.glob("*/coreg_space/img"))
    if direct:
        candidates = [(p.parent.parent.name, p) for p in direct]
    else:
        candidates = [(p.parent.name, p) for p in sorted(root.glob("*/*/coreg_img"))]
    for study_uid, image_dir in candidates:
        if selected is not None and study_uid not in selected:
            continue
        paths = sorted(str(p) for p in image_dir.glob("*.nii.gz"))
        if paths:
            samples.append({
                "study_uid": study_uid,
                "image_paths": paths,
                "n_sequences": len(paths),
            })
    if not samples:
        raise RuntimeError(f"No coregistered NIfTIs found under {root}")
    return samples


def _raw_volume(
    path: str,
    target_shape: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
    posterior_shift_mm: float,
) -> torch.Tensor:
    """Apply MR-RATE's RAS/resample/z-score/crop pipeline to one coreg NIfTI."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(path))
    array = image.get_fdata(dtype=np.float32).transpose(2, 0, 1)
    np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    zooms = image.header.get_zooms()
    current_spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
    new_shape = tuple(
        max(1, int(array.shape[axis] * current_spacing[axis] / target_spacing[axis]))
        for axis in range(3)
    )
    tensor = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(array))[None, None],
        size=new_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    nonzero = tensor != 0
    if bool(nonzero.any()):
        values = tensor[nonzero]
        tensor = (tensor - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    tensor = tensor.clamp(-5, 5) / 5
    out = tensor.new_zeros(target_shape)
    source_slices, target_slices = [], []
    posterior_shift_voxels = int(round(posterior_shift_mm / target_spacing[2]))
    for axis, (have, want) in enumerate(zip(tensor.shape, target_shape)):
        n = min(have, want)
        source_start = max(0, (have - n) // 2)
        if axis == 2 and have > want:
            source_start = min(max(0, source_start - posterior_shift_voxels), have - want)
        target_start = max(0, (want - n) // 2)
        source_slices.append(slice(source_start, source_start + n))
        target_slices.append(slice(target_start, target_start + n))
    out[tuple(target_slices)] = tensor[tuple(source_slices)]
    return out.to(torch.bfloat16)


class MRCoregDINO3DDataset(Dataset):
    """Study-level SSL samples from aligned, variable-count MR sequences."""

    def __init__(
        self,
        *,
        preprocessed_dir: str | None = None,
        data_folder: str | None = None,
        splits_csv: str | None = None,
        split: str = "train",
        space: str = "coreg_space",
        crop_spec: CropSpec = CropSpec(),
        target_spacing: tuple[float, float, float] = (1.0, 0.5, 0.5),
        target_shape: tuple[int, int, int] = (256, 384, 384),
        posterior_shift_mm: float = 15.0,
        cache_files: list[str] | None = None,
        cross_sequence_probability: float = 0.75,
        candidate_trials: int = 12,
        seed: int = 3407,
    ) -> None:
        if (preprocessed_dir is None) == (data_folder is None):
            raise ValueError("Pass exactly one of preprocessed_dir or data_folder")
        if space != "coreg_space":
            raise ValueError("Only coreg_space is valid for aligned cross-sequence MR DINO")
        if not 0 <= cross_sequence_probability <= 1:
            raise ValueError("cross_sequence_probability must be in [0, 1]")
        self.crop_spec = crop_spec
        self.target_shape = tuple(int(x) for x in target_shape)
        self.target_spacing = tuple(float(x) for x in target_spacing)
        self.posterior_shift_mm = float(posterior_shift_mm)
        self.cross_sequence_probability = float(cross_sequence_probability)
        self.candidate_trials = int(candidate_trials)
        self.seed = int(seed)
        self._cached_source: str | None = None
        self._cached_stack: torch.Tensor | None = None
        self._cached_names: list[str] | None = None
        self.preprocessed = preprocessed_dir is not None
        if self.preprocessed:
            manifest = validate_coreg_cache(preprocessed_dir, space, target_spacing)
            self.target_shape = tuple(int(x) for x in manifest["target_shape"])
            if cache_files is None:
                self.samples = discover_coreg_cache(preprocessed_dir, splits_csv, split, space)
            else:
                selected = _load_split_ids(splits_csv, split)
                self.samples = [
                    {
                        "study_uid": Path(path).stem,
                        "cache_path": str(path),
                        "n_sequences": npz_volume_shape(path)[0],
                        "volume_shape": npz_volume_shape(path)[1:],
                    }
                    for path in sorted(cache_files)
                    if selected is None or Path(path).stem in selected
                ]
                if not self.samples:
                    raise RuntimeError("This rank received no coregistered cache studies")
        else:
            selected = _load_split_ids(splits_csv, split)
            self.samples = _discover_raw_coreg(data_folder, selected)
        for sample in self.samples:
            if "volume_shape" in sample and tuple(sample["volume_shape"]) != self.target_shape:
                raise ValueError(
                    f"{sample['cache_path']} shape {sample['volume_shape']} disagrees with "
                    f"manifest {self.target_shape}"
                )
        self.index = [
            (study_index, sequence_index)
            for study_index, sample in enumerate(self.samples)
            for sequence_index in range(int(sample["n_sequences"]))
        ]
        for name, shape in (("global", crop_spec.global_shape), ("local", crop_spec.local_shape)):
            if any(n < 1 for n in shape):
                raise ValueError(f"{name} crop must be positive, got {shape}")

    @property
    def n_studies(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.index)

    def _rng(self, idx: int, epoch: int) -> random.Random:
        value = self.seed + 0x9E3779B1 * int(idx) + 0x85EBCA77 * int(epoch)
        return random.Random(value & ((1 << 63) - 1))

    def _load_stack(self, sample: dict) -> tuple[torch.Tensor, list[str]]:
        source = sample.get("cache_path") or "\0".join(sample.get("image_paths", []))
        if source == self._cached_source:
            return self._cached_stack, self._cached_names
        if self.preprocessed:
            with np.load(sample["cache_path"], allow_pickle=False) as cached:
                if "volumes" not in cached:
                    raise ValueError(f"{sample['cache_path']} has no 'volumes' array")
                array = cached["volumes"]
            if array.ndim != 4 or array.shape[0] < 1:
                raise ValueError(f"Expected [N,D,H,W], got {array.shape} in {sample['cache_path']}")
            if tuple(array.shape[1:]) != self.target_shape:
                raise ValueError(
                    f"Cache tensor shape {array.shape[1:]} disagrees with manifest {self.target_shape}"
                )
            stack = torch.from_numpy(np.ascontiguousarray(array)).to(torch.bfloat16)
            names = [f"sequence_{i:02d}" for i in range(len(stack))]
        else:
            stack = torch.stack([
                _raw_volume(
                    p, self.target_shape, self.target_spacing, self.posterior_shift_mm
                )
                for p in sample["image_paths"]
            ])
            names = [Path(p).name for p in sample["image_paths"]]
        if not bool(torch.isfinite(stack.float()).all()):
            raise ValueError(f"Non-finite voxels in study {sample['study_uid']}")
        self._cached_source = source
        self._cached_stack = stack
        self._cached_names = names
        return stack, names

    def __getitem__(self, key: int | tuple[int, int]) -> dict:
        idx, epoch = key if isinstance(key, tuple) else (key, 0)
        rng = self._rng(idx, epoch)
        study_index, anchor_sequence = self.index[idx]
        sample = self.samples[study_index]
        stack, names = self._load_stack(sample)
        n_sequences = int(stack.shape[0])
        first_sequence = anchor_sequence
        second_sequence = first_sequence
        if n_sequences > 1 and rng.random() < self.cross_sequence_probability:
            second_sequence = rng.randrange(n_sequences - 1)
            if second_sequence >= first_sequence:
                second_sequence += 1
        global_sequences = (first_sequence, second_sequence)

        first_start = _informative_start(
            stack[first_sequence], self.crop_spec.global_shape, rng, self.candidate_trials
        )
        global_starts = _overlapping_global_starts(
            tuple(stack.shape[1:]), self.crop_spec.global_shape, rng, first=first_start
        )
        lo, hi = _intersection_box(global_starts, self.crop_spec.global_shape)

        teacher_globals, student_globals = [], []
        for sequence, start in zip(global_sequences, global_starts):
            clean = _crop_at_start(stack[sequence], self.crop_spec.global_shape, start).unsqueeze(0)
            teacher_globals.append(clean)
            student_globals.append(_student_distortion(clean[0], rng).unsqueeze(0))

        local_views, local_sequences, local_starts = [], [], []
        for _ in range(self.crop_spec.local_crops):
            sequence = rng.randrange(n_sequences)
            start = _contained_start(lo, hi, self.crop_spec.local_shape, rng)
            clean = _crop_at_start(stack[sequence], self.crop_spec.local_shape, start)
            local_views.append(_student_distortion(clean, rng).unsqueeze(0))
            local_sequences.append(sequence)
            local_starts.append(start)

        return {
            "teacher_global": torch.stack(teacher_globals),
            "student_global": torch.stack(student_globals),
            "student_local": torch.stack(local_views),
            "study_uid": sample["study_uid"],
            "sequence_names": names,
            "global_sequences": global_sequences,
            "local_sequences": local_sequences,
            "global_starts": global_starts,
            "local_starts": local_starts,
            "num_sequences": n_sequences,
            "anchor_sequence": anchor_sequence,
            "sample_weight": 1.0 / n_sequences,
            "mask_seed": rng.randrange(2**63 - 1),
        }


def make_block_mask_3d(
    grid: tuple[int, int, int],
    ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate an exact-cardinality block mask on a 3-D patch lattice."""
    total = math.prod(grid)
    target = min(total - 1, max(1, int(round(total * float(ratio)))))
    mask = torch.zeros(grid, dtype=torch.bool)
    tries = 0
    while int(mask.sum()) < target and tries < 64:
        remaining = target - int(mask.sum())
        volume = max(1, min(remaining, int(torch.randint(4, max(5, remaining + 1), (1,), generator=generator))))
        dz = min(grid[0], max(1, round(volume ** (1 / 3) * 0.7)))
        dy = min(grid[1], max(1, round(math.sqrt(volume / dz))))
        dx = min(grid[2], max(1, math.ceil(volume / (dz * dy))))
        z = int(torch.randint(0, grid[0] - dz + 1, (1,), generator=generator))
        y = int(torch.randint(0, grid[1] - dy + 1, (1,), generator=generator))
        x = int(torch.randint(0, grid[2] - dx + 1, (1,), generator=generator))
        mask[z : z + dz, y : y + dy, x : x + dx] = True
        tries += 1
    flat = mask.flatten()
    count = int(flat.sum())
    if count < target:
        choices = (~flat).nonzero().flatten()
        flat[choices[torch.randperm(len(choices), generator=generator)[: target - count]]] = True
    elif count > target:
        choices = flat.nonzero().flatten()
        flat[choices[torch.randperm(len(choices), generator=generator)[: count - target]]] = False
    return flat


def collate_dino3d(
    samples: list[dict],
    patch_size: tuple[int, int, int] = (2, 16, 16),
    mask_ratio: tuple[float, float] = (0.1, 0.5),
) -> dict:
    teacher = torch.stack([sample["teacher_global"] for sample in samples], dim=1)
    student = torch.stack([sample["student_global"] for sample in samples], dim=1)
    local = torch.stack([sample["student_local"] for sample in samples], dim=1)
    _, _, _, d, h, w = student.shape
    grid = tuple(n // p for n, p in zip((d, h, w), patch_size))
    if any(n < 1 for n in grid):
        raise ValueError(f"Global crop {(d, h, w)} is smaller than patch size {patch_size}")
    masks = []
    for crop in range(2):
        for sample in samples:
            generator = torch.Generator().manual_seed(int(sample["mask_seed"]) + crop * 1_000_003)
            ratio = float(torch.empty(1).uniform_(*mask_ratio, generator=generator))
            masks.append(make_block_mask_3d(grid, ratio, generator))
    masks = torch.stack(masks)
    indices = masks.flatten().nonzero().flatten()
    weights = (1.0 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return {
        "teacher_global": teacher,
        "student_global": student,
        "student_local": local,
        "masks": masks,
        "mask_indices": indices,
        "mask_weights": weights,
        "n_masked": torch.tensor([len(indices)], dtype=torch.long),
        "sample_weights": torch.tensor([float(s["sample_weight"]) for s in samples]),
        "metadata": [
            {k: sample[k] for k in (
                "study_uid", "sequence_names", "global_sequences", "local_sequences",
                "global_starts", "local_starts", "num_sequences", "anchor_sequence",
            )}
            for sample in samples
        ],
    }


def stage_crop_spec(stage: str) -> CropSpec:
    if stage in {"pretrain", "gram"}:
        return CropSpec(global_shape=(64, 192, 192), local_shape=(32, 96, 96), local_crops=8)
    if stage == "highres":
        return CropSpec(global_shape=(64, 384, 384), local_shape=(32, 192, 192), local_crops=4)
    raise ValueError(f"Unknown stage: {stage}")
