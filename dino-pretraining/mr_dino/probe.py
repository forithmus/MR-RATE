"""Validation MIL probe for MR DINO checkpoints (the FORA CT-DINO checkpoint protocol).

A checkpoint is scored without touching train or test data:

1. ``export``: copy the EMA-teacher backbone out of a DCP training checkpoint into
   one small ``.pt`` file, so later steps do not depend on checkpoint rotation.
2. ``extract``: run the frozen backbone over every sequence of every labelled
   study of one split (default ``val``).
   - Each volume goes through the training preprocessing (same space, spacing and
     shape as the checkpoint), cut into training-size tiles (default: the phase-1
     global crop, 64x192x192 voxels).
   - Output: the last block's normalised patch tokens.
   - Tokens are average-pooled ``--pool`` cells (default 2x2x2) over head-foreground
     tokens. Air-only cells are dropped.
   - Each worker writes one raw fp16 token file plus one index, so the inode use is
     tiny.
3. ``cv``: patient-level K-fold cross-validation of NeuroVFM's ClassifyThenAggregate
   head (the MR-RATE MIL head) on the frozen bags. One bag holds all sequences of a
   study. Only out-of-fold predictions are kept.
4. ``summarize``: per-class and macro AUROC / AUPRC over all studies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    ALIGNED_SPACES,
    _load_split_ids,
    _raw_volume,
    discover_raw_aligned,
)

TEACHER_PREFIX = "model.teacher.backbone."
EXPORT_FORMAT = "mrdino_teacher_backbone_v1"
INDEX_FORMAT = "mrdino_probe_tokens_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "contrastive-pretraining/scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv"
DEFAULT_SPLITS = REPO_ROOT / "contrastive-pretraining/scripts/eval_labels/splits_merged_majority/splits.csv"


# --------------------------------------------------------------------------- checkpoint

def read_checkpoint_metadata(checkpoint: Path) -> dict:
    """Training args/step of a DCP checkpoint directory or an exported teacher file."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        meta = json.loads((checkpoint / "metadata.json").read_text())
        return {"step": int(meta["step"]), "stage": meta["stage"], "args": meta["args"], "source": str(checkpoint)}
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{checkpoint} is neither a DCP checkpoint directory nor a {EXPORT_FORMAT} file")
    return {k: state[k] for k in ("step", "stage", "args", "source")}


def load_teacher_backbone(checkpoint: Path, device: torch.device | str = "cpu"):
    """Rebuild the EMA-teacher backbone from a DCP checkpoint (no process group needed) or an export."""
    from .train_ddp import build_backbone

    checkpoint = Path(checkpoint)
    meta = read_checkpoint_metadata(checkpoint)
    backbone = build_backbone(meta["args"]["arch"])
    if checkpoint.is_dir():
        if not (checkpoint / "COMPLETE").exists():
            raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
        import torch.distributed.checkpoint as dcp

        wanted = {TEACHER_PREFIX + k: torch.empty_like(v) for k, v in backbone.state_dict().items()}
        dcp.load(wanted, storage_reader=dcp.FileSystemReader(str(checkpoint)), no_dist=True)
        state = {k[len(TEACHER_PREFIX):]: v for k, v in wanted.items()}
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    backbone.load_state_dict(state, strict=True)
    return backbone.to(device).eval().requires_grad_(False), meta


def export_teacher(checkpoint: Path, out: Path) -> Path:
    backbone, meta = load_teacher_backbone(checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    torch.save({"format": EXPORT_FORMAT, **meta, "state_dict": backbone.state_dict()}, tmp)
    os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------- tokens

def _pad_to_multiple(x: torch.Tensor, multiple: tuple[int, int, int], value: float) -> torch.Tensor:
    pads = []
    for n, m in reversed(list(zip(x.shape[-3:], multiple))):
        pads.extend((0, (-n) % m))
    return F.pad(x, pads, value=value) if any(pads) else x


@torch.inference_mode()
def embed_sequence(
    backbone,
    volume: torch.Tensor,
    tile: tuple[int, int, int],
    pool: tuple[int, int, int],
    patch: tuple[int, int, int] = (2, 16, 16),
    fg_margin: float = 0.05,
    tiles_per_batch: int = 8,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pooled foreground tokens ``[M, C]`` (fp16) and their pooled-grid coords ``[M, 3]``.

    Background is the volume minimum (air after MR-RATE's z-score; padding uses it too).
    A token counts as foreground when some voxel of its patch is ``fg_margin`` above it.
    """
    if any(t % p for t, p in zip(tile, patch)):
        raise ValueError(f"Tile {tile} must be a multiple of the patch {patch}")
    device = torch.device(device)
    background = float(volume.float().min())
    vol = _pad_to_multiple(volume.float(), tile, background)
    grid = tuple(n // p for n, p in zip(vol.shape, patch))
    tile_grid = tuple(t // p for t, p in zip(tile, patch))
    starts = [
        (z, y, x)
        for z in range(0, vol.shape[0], tile[0])
        for y in range(0, vol.shape[1], tile[1])
        for x in range(0, vol.shape[2], tile[2])
    ]
    feats = torch.zeros(*grid, backbone.embed_dim, device=device, dtype=torch.float32)
    fg = torch.zeros(grid, device=device, dtype=torch.bool)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    for i in range(0, len(starts), tiles_per_batch):
        chunk = starts[i : i + tiles_per_batch]
        x = torch.stack([vol[z : z + tile[0], y : y + tile[1], xx : xx + tile[2]] for z, y, xx in chunk])
        x = x.to(device)[:, None]
        tile_fg = F.max_pool3d(x - background, kernel_size=patch, stride=patch)[:, 0] > fg_margin
        keep = tile_fg.flatten(1).any(1)
        if not bool(keep.any()):
            continue
        with autocast:
            out = backbone.forward_features(x[keep].to(torch.bfloat16 if device.type == "cuda" else torch.float32))
        tokens = out["x_norm_patchtokens"].float().reshape(-1, *tile_grid, backbone.embed_dim)
        for (z, y, xx), t, m in zip([s for s, k in zip(chunk, keep.tolist()) if k], tokens, tile_fg[keep]):
            gz, gy, gx = z // patch[0], y // patch[1], xx // patch[2]
            feats[gz : gz + tile_grid[0], gy : gy + tile_grid[1], gx : gx + tile_grid[2]] = t
            fg[gz : gz + tile_grid[0], gy : gy + tile_grid[1], gx : gx + tile_grid[2]] = m
    # foreground-weighted average pooling on the token grid
    weight = _pad_to_multiple(fg.float()[None], pool, 0.0)[0]
    feats = _pad_to_multiple(feats.permute(3, 0, 1, 2), pool, 0.0)
    count = F.avg_pool3d(weight[None, None], pool, pool)[0, 0]
    summed = F.avg_pool3d((feats * weight)[None], pool, pool)[0]
    cells = count > 0
    pooled = (summed[:, cells] / count[cells]).T
    coords = cells.nonzero().to(torch.int16)
    return pooled.to(torch.float16).cpu(), coords.cpu()


def worker_identity() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return rank, world


def read_labels(labels_csv: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with open(labels_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        names = [c for c in reader.fieldnames if c not in {"study_uid", "subject_id"}]
        key = "study_uid" if "study_uid" in reader.fieldnames else "subject_id"
        labels = {row[key].strip(): np.array([float(row[c]) for c in names], dtype=np.float32) for row in reader}
    return names, labels


def probe_studies(args) -> list[dict]:
    """Labelled studies of the split, sorted by uid (identical on every worker)."""
    _, labels = read_labels(Path(args.labels_csv))
    selected = _load_split_ids(str(args.splits_csv), args.split)
    selected = {uid for uid in selected if uid in labels}
    studies = sorted(discover_raw_aligned(args.data_folder, selected, args.space), key=lambda s: s["study_uid"])
    if args.max_studies:
        studies = studies[: args.max_studies]
    if not studies:
        raise RuntimeError(f"No labelled {args.split} studies under {args.data_folder}")
    return studies


class _StudyVolumes(torch.utils.data.Dataset):
    def __init__(self, studies, target_shape, target_spacing, posterior_shift_mm):
        self.studies, self.shape, self.spacing, self.shift = studies, target_shape, target_spacing, posterior_shift_mm

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        study = self.studies[i]
        volumes = [_raw_volume(p, self.shape, self.spacing, self.shift) for p in study["image_paths"]]
        return study["study_uid"], [Path(p).name for p in study["image_paths"]], volumes


def extract(args) -> Path:
    rank, world = worker_identity()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone, meta = load_teacher_backbone(Path(args.checkpoint), device)
    train = meta["args"]
    args.space = args.space or train.get("space", "atlas_space")   # pre-coreg checkpoints were atlas
    if args.space != train.get("space", "atlas_space") and not args.allow_space_mismatch:
        raise ValueError(f"Checkpoint was trained on {train.get('space')!r}, probe asked for {args.space!r}")
    target_shape = tuple(train["target_shape"])
    target_spacing = tuple(train["target_spacing"])
    out = Path(args.features_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / f"index_rank{rank:04d}.npz"
    if index_path.exists():
        print(f"[probe] rank {rank}: {index_path} exists, skipping", flush=True)
        return index_path
    studies = probe_studies(args)[rank::world]
    loader = torch.utils.data.DataLoader(
        _StudyVolumes(studies, target_shape, target_spacing, float(train["posterior_shift_mm"])),
        batch_size=None, shuffle=False, num_workers=args.workers,
        prefetch_factor=1 if args.workers else None, persistent_workers=False,
    )
    token_path = out / f"tokens_rank{rank:04d}.f16"
    uids, starts, counts, seq_ids, coords, names = [], [], [], [], [], []
    total, t0 = 0, time.time()
    with open(token_path, "wb") as sink:
        for i, (uid, seq_names, volumes) in enumerate(loader):
            start = total
            for s, volume in enumerate(volumes):
                tokens, xyz = embed_sequence(
                    backbone, volume, tuple(args.tile), tuple(args.pool),
                    fg_margin=args.fg_margin, tiles_per_batch=args.tiles_per_batch, device=device,
                )
                sink.write(tokens.numpy().tobytes())
                total += len(tokens)
                seq_ids.append(np.full(len(tokens), s, dtype=np.int16))
                coords.append(xyz.numpy())
            if total == start:
                raise RuntimeError(f"Study {uid} produced no foreground tokens")
            uids.append(uid)
            starts.append(start)
            counts.append(total - start)
            names.append(json.dumps(seq_names))
            if rank == 0 and (i % 20 == 0 or i + 1 == len(studies)):
                rate = (i + 1) / max(1e-6, time.time() - t0)
                print(f"[probe] rank 0: {i + 1}/{len(studies)} studies, {total:,} tokens, {rate:.2f} studies/s", flush=True)
    tmp = out / f"index_rank{rank:04d}.partial.npz"
    np.savez(
        tmp,
        format=INDEX_FORMAT, dim=backbone.embed_dim, token_file=token_path.name,
        study_uid=np.array(uids), start=np.array(starts, dtype=np.int64), count=np.array(counts, dtype=np.int64),
        seq_id=np.concatenate(seq_ids) if seq_ids else np.zeros(0, np.int16),
        coords=np.concatenate(coords) if coords else np.zeros((0, 3), np.int16),
        sequence_names=np.array(names),
        checkpoint_step=meta["step"], checkpoint_stage=meta["stage"], checkpoint_source=meta["source"],
        space=args.space, tile=np.array(args.tile), pool=np.array(args.pool),
    )
    os.replace(tmp, index_path)
    return index_path


# --------------------------------------------------------------------------- MIL head

def mil_head_class():
    """NeuroVFM ClassifyThenAggregate from MR-RATE's MIL probe (single source of truth)."""
    path = Path(os.environ.get("MRRATE_MIL_PROBE_MODULE", REPO_ROOT / "contrastive-pretraining/scripts/mil_probe.py"))
    spec = importlib.util.spec_from_file_location("mrrate_mil_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClassifyThenAggregate


class TokenBags:
    """All workers' token files as memory maps, one bag per study."""

    def __init__(self, features_dir: Path):
        self.uids, self.bags, self.dim = [], [], None
        indexes = sorted(Path(features_dir).glob("index_rank*.npz"))
        if not indexes:
            raise FileNotFoundError(f"No index_rank*.npz under {features_dir}")
        self.meta = {}
        for path in indexes:
            with np.load(path, allow_pickle=False) as index:
                if str(index["format"]) != INDEX_FORMAT:
                    raise ValueError(f"{path} is not a {INDEX_FORMAT} index")
                dim = int(index["dim"])
                if self.dim not in (None, dim):
                    raise ValueError("Workers disagree on the token width")
                self.dim = dim
                tokens = np.memmap(path.parent / str(index["token_file"]), dtype=np.float16, mode="r").reshape(-1, dim)
                for uid, start, count in zip(index["study_uid"], index["start"], index["count"]):
                    self.uids.append(str(uid))
                    self.bags.append((tokens, int(start), int(count)))
                self.meta = {k: index[k].tolist() for k in ("checkpoint_step", "checkpoint_stage", "space", "tile", "pool")}
        if len(set(self.uids)) != len(self.uids):
            raise ValueError("A study appears in more than one worker shard")

    def __len__(self):
        return len(self.uids)

    def tokens(self, i: int) -> np.ndarray:
        array, start, count = self.bags[i]
        return np.asarray(array[start : start + count])


def patient_folds(uids: list[str], patient_of: dict[str, str], k: int, seed: int) -> np.ndarray:
    """Deterministic patient-level fold id per study (all studies of a patient share a fold)."""
    patients = sorted({patient_of.get(uid, uid) for uid in uids})
    order = sorted(patients, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
    fold_of = {p: i % k for i, p in enumerate(order)}
    return np.array([fold_of[patient_of.get(uid, uid)] for uid in uids])


def read_patients(splits_csv: Path) -> dict[str, str]:
    with open(splits_csv, newline="") as handle:
        return {row["study_uid"].strip(): row.get("patient_uid", row["study_uid"]).strip() for row in csv.DictReader(handle)}


def _batch(bags: TokenBags, items, device):
    arrays = [bags.tokens(i) for i in items]
    lengths = torch.tensor([0] + [len(a) for a in arrays])
    tokens = torch.from_numpy(np.concatenate(arrays)).to(device, non_blocking=True)
    return tokens, torch.cumsum(lengths, 0).to(device)


def run_fold(args) -> Path:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bags = TokenBags(Path(args.features_dir))
    names, labels = read_labels(Path(args.labels_csv))
    y = np.stack([labels[uid] for uid in bags.uids])
    folds = patient_folds(bags.uids, read_patients(Path(args.splits_csv)), args.folds, args.seed)
    train = np.flatnonzero(folds != args.fold)
    held = np.flatnonzero(folds == args.fold)
    torch.manual_seed(args.seed + args.fold)
    rng = np.random.default_rng(args.seed + args.fold)

    head = mil_head_class()(bags.dim, len(names), hidden_dim=args.hidden_dim, mlp_hidden_dims=(args.mlp_hidden_dim,)).to(device)
    positives = y[train].sum(0)
    pos_weight = np.clip((len(train) - positives) / np.clip(positives, 1.0, None), 1.0, 100.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    warmup = max(1, int(0.1 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    head.train()
    for epoch in range(args.epochs):
        order = rng.permutation(train)
        for b in range(0, len(order), args.batch_size):
            items = order[b : b + args.batch_size]
            tokens, cu = _batch(bags, items, device)
            with autocast:
                logits = head(tokens.float(), cu)
            loss = loss_fn(logits.float(), torch.from_numpy(y[items]).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            schedule.step()
        print(f"[probe] fold {args.fold} epoch {epoch + 1}/{args.epochs} loss {float(loss.detach()):.4f}", flush=True)
    head.eval()
    out_logits = []
    with torch.inference_mode():
        for b in range(0, len(held), args.batch_size):
            items = held[b : b + args.batch_size]
            tokens, cu = _batch(bags, items, device)
            with autocast:
                out_logits.append(head(tokens.float(), cu).float().cpu().numpy())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"oof_fold{args.fold}.npz"
    np.savez(path, study_uid=np.array([bags.uids[i] for i in held]), logits=np.concatenate(out_logits), labels=y[held])
    return path


# --------------------------------------------------------------------------- metrics

def auroc(target: np.ndarray, score: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties."""
    positive = target == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    if not (target == 1).any():
        return None
    hits = target[np.argsort(-score, kind="mergesort")] == 1
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def summarize(args) -> dict:
    out = Path(args.out_dir)
    names, _ = read_labels(Path(args.labels_csv))
    parts = [np.load(out / f"oof_fold{f}.npz") for f in range(args.folds)]
    uids = np.concatenate([p["study_uid"] for p in parts])
    if len(set(uids.tolist())) != len(uids):
        raise ValueError("A study was predicted by more than one fold")
    logits = np.concatenate([p["logits"] for p in parts])
    labels = np.concatenate([p["labels"] for p in parts])
    rows = []
    for c, name in enumerate(names):
        rows.append({
            "label": name, "positives": int(labels[:, c].sum()), "studies": int(len(labels)),
            "auroc": auroc(labels[:, c], logits[:, c]), "auprc": average_precision(labels[:, c], logits[:, c]),
        })
    valid = [r for r in rows if r["auroc"] is not None]
    meta = TokenBags(Path(args.features_dir)).meta if args.features_dir and Path(args.features_dir).exists() else {}
    fold_auroc = []
    for p in parts:
        per = [auroc(p["labels"][:, c], p["logits"][:, c]) for c in range(len(names))]
        per = [v for v in per if v is not None]
        fold_auroc.append(float(np.mean(per)) if per else None)
    result = {
        "macro_auroc": float(np.mean([r["auroc"] for r in valid])),
        "macro_auprc": float(np.mean([r["auprc"] for r in valid])),
        "fold_macro_auroc": fold_auroc,
        "studies": int(len(labels)),
        "folds": args.folds,
        "labels_csv": str(args.labels_csv),
        "per_class": rows,
        **meta,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_class.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"[probe] step {meta.get('checkpoint_step', '?')}: macro AUROC {result['macro_auroc']:.4f} "
        f"AUPRC {result['macro_auprc']:.4f} over {result['studies']} studies "
        f"(folds {', '.join(f'{v:.3f}' for v in fold_auroc if v is not None)})",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- CLI

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="Copy the EMA-teacher backbone out of a DCP checkpoint")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--labels-csv", default=str(DEFAULT_LABELS), help="study_uid + binary label columns (default: 14 merged groups)")
    common.add_argument("--splits-csv", default=str(DEFAULT_SPLITS), help="study_uid, patient_uid, split")
    common.add_argument("--features-dir", required=True)
    common.add_argument("--device", default=None)

    x = sub.add_parser("extract", parents=[common], help="Frozen tokens of every labelled study of a split")
    x.add_argument("--checkpoint", required=True, help="DCP checkpoint directory or exported teacher .pt")
    x.add_argument("--data-folder", required=True, help="Extracted MR-RATE tree: batchXX/<study>/coreg_img (or atlas_img)")
    x.add_argument("--space", default=None, choices=ALIGNED_SPACES, help="Default: the checkpoint's training space")
    x.add_argument("--allow-space-mismatch", action="store_true")
    x.add_argument("--split", default="val")
    x.add_argument("--tile", type=int, nargs=3, default=(64, 192, 192), help="Voxels; default = phase-1 global crop")
    x.add_argument("--pool", type=int, nargs=3, default=(2, 2, 2), help="Token-grid pooling cell")
    x.add_argument("--fg-margin", type=float, default=0.05, help="Foreground: patch max above the volume minimum")
    x.add_argument("--tiles-per-batch", type=int, default=8)
    x.add_argument("--workers", type=int, default=6)
    x.add_argument("--max-studies", type=int, default=None, help="Smoke only")

    c = sub.add_parser("cv", parents=[common], help="Train/predict one patient-level fold")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--fold", type=int, required=True)
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--epochs", type=int, default=8)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--lr", type=float, default=2e-3)
    c.add_argument("--weight-decay", type=float, default=0.05)
    c.add_argument("--hidden-dim", type=int, default=512)
    c.add_argument("--mlp-hidden-dim", type=int, default=384)

    s = sub.add_parser("summarize", parents=[common], help="Out-of-fold macro/per-class AUROC and AUPRC")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--folds", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        print(export_teacher(Path(args.checkpoint), Path(args.out)))
    elif args.command == "extract":
        print(extract(args))
    elif args.command == "cv":
        print(run_fold(args))
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
