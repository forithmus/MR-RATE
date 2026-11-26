"""Online MRI study features for MIL, with the outputs of ``mr_dino.vision.online_study_factory``.

* The frozen prefix (all blocks but the last) runs once per study, tiles batched, and its
  tokens stay on the GPU so both MIL passes reuse them instead of recomputing.
* Volumes for upcoming studies are read and resampled in background threads.

Three input layouts:

* ``cached_study_factory``: every sequence is a one-channel input and every token of every
  sequence enters the bag (the MR-RATE MIL).
* the same function with a modality-channel release (``config["input_channels"]``): each
  input stacks one acquisition per modality plus availability planes (``mr_dino.channels``).
  Stack ``k`` uses acquisition ``k mod n_m`` of each modality, so every acquisition is seen.
* ``cross_sequence_factory``: per tile position, the tokens of all sequences at each atlas
  position, for ``fusion.CrossSequenceAttention``.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mr_dino.channels import acquisition_stacks, modality_groups, stack_channels
from mr_dino.data import raw_volume_with_support
from mr_dino.geometry import normalized_positions, sequence_kind

RANK = int(os.environ.get("RANK", 0))
TIMING_LOG = os.environ.get("MIL_TIMING_LOG")


def emit(**row) -> None:
    if not TIMING_LOG:
        return
    row["rank"] = RANK
    with open(f"{TIMING_LOG}.rank{RANK}", "a") as handle:
        handle.write(json.dumps(row) + "\n")


def modality_layout(metadata) -> bool:
    layout = metadata["config"].get("input_channels")
    if layout is None:
        return False
    if layout.get("layout") != "modality":
        raise ValueError(f"Unknown input channel layout {layout}")
    return True


def load_study(study, metadata):
    """One ``(image, support)`` per sequence path; ``None`` for paths a modality release ignores."""
    args = metadata["args"]
    start = time.perf_counter()
    used = set(range(len(study["image_paths"])))
    if modality_layout(metadata):
        used = {i for group in modality_groups(study["image_paths"]) for i in group}
    volumes = [raw_volume_with_support(source, tuple(args["target_shape"]), tuple(args["target_spacing"]),
                                       args["posterior_shift_mm"]) if i in used else None
               for i, source in enumerate(study["image_paths"])]
    emit(event="load", uid=study["study_uid"], sequences=len(used), seconds=time.perf_counter() - start)
    return volumes


class Prefetcher:
    """Loads items ``i+1 .. i+depth`` while item ``i`` is being used; ``None`` entries are skipped."""

    def __init__(self, keys, load, depth=2):
        self.keys, self.load, self.depth = keys, load, depth
        self.pool = ThreadPoolExecutor(max_workers=max(1, depth))
        self.futures = {}

    def _submit(self, i):
        if 0 <= i < len(self.keys) and self.keys[i] is not None and i not in self.futures:
            self.futures[i] = self.pool.submit(self.load, self.keys[i])

    def get(self, i):
        for j in range(i, i + self.depth + 1):
            self._submit(j)
        start = time.perf_counter()
        out = self.futures.pop(i).result()
        emit(event="load_wait", seconds=time.perf_counter() - start)
        return out

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def _padded(image, support, tile):
    """Pad ``[C, D, H, W]`` image and ``[D, H, W]`` support to whole tiles."""
    pads = []
    for n, t in reversed(list(zip(image.shape[-3:], tile))):
        pads.extend((0, (-n) % t))
    return F.pad(image, pads, value=0), F.pad(support.float(), pads, value=0)


def _tile_starts(shape, tile):
    for z in range(0, shape[0], tile[0]):
        for y in range(0, shape[1], tile[1]):
            for x in range(0, shape[2], tile[2]):
                yield z, y, x


def _inputs(study, volumes, metadata):
    """``(image [C, D, H, W], support [D, H, W], kind)`` per backbone input."""
    if not modality_layout(metadata):
        for sequence, (volume, support) in enumerate(volumes):
            yield volume[None], support, sequence_kind(study["image_paths"][sequence])
        return
    for stack in acquisition_stacks(modality_groups(study["image_paths"])):
        images = [None if i is None else volumes[i][0] for i in stack]
        support = torch.zeros_like(next(volumes[i][1] for i in stack if i is not None))
        for i in stack:
            if i is not None:
                support |= volumes[i][1]
        yield stack_channels(images), support, -1


def _encode(backbone, crops, device, tile_batch):
    """Frozen prefix tokens of each crop ``[C, d, h, w]``, kept on the GPU."""
    out = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for i in range(0, len(crops), tile_batch):
            crop = torch.stack(crops[i:i + tile_batch]).to(device, non_blocking=True)
            tokens, grid = backbone.prepare_tokens_with_masks(crop)
            rope = tuple(r.to(device=tokens.device, dtype=tokens.dtype)
                         for r in backbone.rope_embed(D=grid[0], H=grid[1], W=grid[2]))
            for block in backbone.blocks[:-1]:
                tokens = block(tokens, rope)
            out.extend((tokens[j:j + 1], grid) for j in range(len(tokens)))
    return out


def cached_study_factory(study, volumes, backbone, adapter, metadata, device, tile_batch=4):
    tile = tuple(metadata["config"]["training_crops"]["global"])
    patch = tuple(metadata["config"]["backbone"]["patch_size"])
    in_chans = int(metadata["config"]["backbone"].get("in_chans", 1))
    tiles = []
    for image, support, kind in _inputs(study, volumes, metadata):
        if image.shape[0] != in_chans:
            raise ValueError(f"Backbone takes {in_chans} channels, input has {image.shape[0]}")
        image, valid = _padded(image, support, tile)
        global_grid = tuple(n // p for n, p in zip(image.shape[-3:], patch))
        for z, y, x in _tile_starts(image.shape[-3:], tile):
            sl = (slice(z, z + tile[0]), slice(y, y + tile[1]), slice(x, x + tile[2]))
            keep = F.max_pool3d(valid[sl][None, None], patch, patch).flatten() > 0
            if not bool(keep.any()):
                continue
            start = tuple(a // p for a, p in zip((z, y, x), patch))
            tiles.append((image[(slice(None),) + sl], keep, start, global_grid, kind))
    prefix = []

    def factory():
        if not prefix:
            prefix.extend(_encode(backbone, [t[0] for t in tiles], device, tile_batch))
        for (_, keep, start, global_grid, kind), (tokens, grid) in zip(tiles, prefix):
            keep = keep.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                features = adapter(tokens, grid)[0, keep]
            coordinates = normalized_positions(start, grid, global_grid, device)[keep]
            yield features.float(), None, coordinates, kind

    return factory


def cross_sequence_factory(study, volumes, backbone, adapter, metadata, device, tile_batch=4, recompute=True):
    """Per tile position: tokens ``[N, S, D]`` of the ``S`` sequences that cover it, ``valid [N, S]``,
    sequence kinds ``[S]`` and coordinates ``[N, 3]`` of the ``N`` positions covered by any sequence.

    ``recompute`` checkpoints the trainable last block per sequence, so the backward pass holds
    one sequence's activations at a time.
    """
    if modality_layout(metadata):
        raise ValueError("Cross-sequence attention takes one-channel sequence inputs")
    tile = tuple(metadata["config"]["training_crops"]["global"])
    patch = tuple(metadata["config"]["backbone"]["patch_size"])
    padded = []
    for sequence, (volume, support) in enumerate(volumes):
        image, valid = _padded(volume[None], support, tile)
        padded.append((image, valid, sequence_kind(study["image_paths"][sequence])))
    shapes = {tuple(image.shape[-3:]) for image, _, _ in padded}
    if len(shapes) != 1:
        raise ValueError(f"Sequences of {study['study_uid']} are not on one grid: {sorted(shapes)}")
    shape = shapes.pop()
    global_grid = tuple(n // p for n, p in zip(shape, patch))
    positions, crops = [], []
    for z, y, x in _tile_starts(shape, tile):
        sl = (slice(z, z + tile[0]), slice(y, y + tile[1]), slice(x, x + tile[2]))
        members = []
        for image, valid, kind in padded:
            keep = F.max_pool3d(valid[sl][None, None], patch, patch).flatten() > 0
            if bool(keep.any()):
                members.append((len(crops), keep, kind))
                crops.append(image[(slice(None),) + sl])
        if members:
            positions.append((tuple(a // p for a, p in zip((z, y, x), patch)), members))
    prefix = []

    def last_block(tokens, grid):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return adapter(tokens, grid)[0].float()

    def factory():
        if not prefix:
            prefix.extend(_encode(backbone, crops, device, tile_batch))
        for start, members in positions:
            grid = prefix[members[0][0]][1]
            outputs = []
            for index, _, _ in members:
                tokens = prefix[index][0]
                if recompute and torch.is_grad_enabled():
                    outputs.append(checkpoint(last_block, tokens, grid, use_reentrant=False))
                else:
                    outputs.append(last_block(tokens, grid))
            tokens = torch.stack(outputs, 1)
            valid = torch.stack([keep for _, keep, _ in members], 1).to(device)
            covered = valid.any(1)
            kinds = torch.tensor([kind for _, _, kind in members], device=device)
            coordinates = normalized_positions(start, grid, global_grid, device)[covered]
            yield tokens[covered], valid[covered], kinds, coordinates

    return factory


class Deadline:
    """Collective stop decision so every rank checkpoints and exits at the same step."""

    def __init__(self, device, margin_seconds=0.0):
        epoch = float(os.environ.get("MIL_DEADLINE_EPOCH", "0") or 0)
        self.at = epoch - margin_seconds if epoch else None
        self.device = device

    def remaining(self):
        return float("inf") if self.at is None else self.at - time.time()

    def reached(self, reserve_seconds=0.0):
        flag = torch.tensor(float(self.remaining() < reserve_seconds), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
