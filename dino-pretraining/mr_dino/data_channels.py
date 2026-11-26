"""Multi-channel DINO samples: the MR modalities of a study are the channels of one input.

One item per study. Each item draws one acquisition per modality (T1, T2, FLAIR, SWI when
present) and builds a ``[2M, D, H, W]`` stack (``mr_dino.channels``). Every view keeps a
random nonempty subset of the available modalities (the full set with probability
``full_probability``); dropped modalities are zero with availability 0. The two global views
use the same subset unless, with ``cross_sequence_probability``, the second draws its own.
iBOT/cross-crop twin pairs follow ``global_sequences``, which holds each global view's subset
code, so with ``cross_view_sequences="same"`` only views with identical inputs are paired.
"""
from __future__ import annotations

import random

import torch

from .channels import MODALITIES, drop_channels, modality_groups, random_subset, stack_channels, subset_code
from .data import (
    MRAlignedDINO3DDataset,
    _contained_start,
    _crop_at_start,
    _informative_start,
    _overlapping_global_starts,
    _raw_volume,
    _student_distortion,
    snap_second_start,
)


def _crop_channels(stack: torch.Tensor, shape, start) -> torch.Tensor:
    return torch.stack([_crop_at_start(channel, shape, start) for channel in stack])


def _distort_channels(view: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Independent MR-safe intensity distortion per present image channel; planes unchanged."""
    m = view.shape[0] // 2
    out = view.clone()
    for channel in range(m):
        if bool(view[m + channel].any()):
            out[channel] = _student_distortion(view[channel], rng)
    return out


class MRMultiChannelDINO3DDataset(MRAlignedDINO3DDataset):
    def __init__(self, *, full_probability: float = 0.5, **kwargs) -> None:
        if kwargs.get("preprocessed_dir") is not None:
            raise ValueError("Modality channels need sequence names; use the raw NIfTI path (--data-folder)")
        super().__init__(**kwargs)
        if not 0 <= full_probability <= 1:
            raise ValueError("full_probability must be in [0, 1]")
        self.full_probability = float(full_probability)
        kept = []
        for sample in self.samples:
            groups = modality_groups(sample["image_paths"])
            if any(groups):
                kept.append({**sample, "modality_groups": groups})
        if not kept:
            raise RuntimeError("No study has an acquisition of any input modality")
        self.samples = kept
        self.index = [(i, 0) for i in range(len(self.samples))]
        self.in_chans = 2 * len(MODALITIES)

    def _load_choice(self, sample: dict, rng: random.Random) -> tuple[torch.Tensor, list[str | None]]:
        paths = [sample["image_paths"][rng.choice(g)] if g else None for g in sample["modality_groups"]]
        volumes = [None if p is None else
                   _raw_volume(p, self.target_shape, self.target_spacing, self.posterior_shift_mm) for p in paths]
        stack = stack_channels(volumes, self.target_shape)
        if not bool(torch.isfinite(stack.float()).all()):
            raise ValueError(f"Non-finite voxels in study {sample['study_uid']}")
        return stack, paths

    def __getitem__(self, key) -> dict:
        idx, epoch = key if isinstance(key, tuple) else (key, 0)
        rng = self._rng(idx, epoch)
        sample = self.samples[idx]
        stack, paths = self._load_choice(sample, rng)
        m = len(MODALITIES)
        available = [c for c in range(m) if paths[c] is not None]

        first_subset = random_subset(available, rng, self.full_probability)
        second_subset = first_subset
        if len(available) > 1 and rng.random() < self.cross_sequence_probability:
            second_subset = random_subset(available, rng, self.full_probability)
        subsets = (first_subset, second_subset)

        reference = stack[:m].float().abs().sum(0)
        first_start = _informative_start(reference, self.crop_spec.global_shape, rng, self.candidate_trials)
        container = tuple(stack.shape[1:])
        first, second = _overlapping_global_starts(container, self.crop_spec.global_shape, rng,
                                                   first=first_start, min_overlap=self.global_overlap)
        second = snap_second_start(first, second, container, self.crop_spec.global_shape, self.patch_size)
        global_starts = (first, second)

        teacher_globals, student_globals = [], []
        for keep, start in zip(subsets, global_starts):
            clean = drop_channels(_crop_channels(stack, self.crop_spec.global_shape, start), keep)
            teacher_globals.append(clean.to(torch.bfloat16))
            student_globals.append(_distort_channels(clean, rng).to(torch.bfloat16))

        local_views, local_codes, local_starts = [], [], []
        for _ in range(self.crop_spec.local_crops):
            keep = random_subset(available, rng, self.full_probability)
            parent = global_starts[rng.randrange(2)]
            hi = tuple(min(p + g, c) for p, g, c in zip(parent, self.crop_spec.global_shape, container))
            start = _contained_start(parent, hi, self.crop_spec.local_shape, rng)
            clean = drop_channels(_crop_channels(stack, self.crop_spec.local_shape, start), keep)
            local_views.append(_distort_channels(clean, rng).to(torch.bfloat16))
            local_codes.append(subset_code(keep))
            local_starts.append(start)

        return {
            "teacher_global": torch.stack(teacher_globals),
            "student_global": torch.stack(student_globals),
            "student_local": torch.stack(local_views),
            "study_uid": sample["study_uid"],
            "sequence_names": [p or "" for p in paths],
            "global_sequences": tuple(subset_code(s) for s in subsets),
            "local_sequences": local_codes,
            "global_starts": global_starts,
            "local_starts": local_starts,
            "num_sequences": len(available),
            "anchor_sequence": 0,
            "sample_weight": 1.0,
            "mask_seed": rng.randrange(2**63 - 1),
        }
