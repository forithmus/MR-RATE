"""Distributed, resumable 3-D DINOv3 training for atlas-registered MR-RATE."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .data import (
    MRAtlasDINO3DDataset,
    InfiniteStudySampler,
    collate_dino3d,
    discover_atlas_cache,
    discover_raw_atlas_split,
    npz_volume_shape,
    stage_crop_spec,
)
from .model import DinoVisionTransformer3D, vit_7b_3d, vit_hplus_3d, vit_large_3d
from .objective import DINO3DLearner, LossWeights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--preprocessed-dir", help="MR-RATE cache root containing atlas_space/*.npz")
    source.add_argument(
        "--data-folder",
        help="MR-RATE-atlas NIfTI tree; uses the same live loader as previous MIL training",
    )
    p.add_argument("--splits-csv")
    p.add_argument("--split", default="train")
    p.add_argument("--space", default="atlas_space", choices=("atlas_space",))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=("pretrain", "gram", "highres"), default="pretrain")
    p.add_argument("--arch", choices=("tiny", "large", "hplus"), default="hplus")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--prototypes", type=int, default=65_536)
    p.add_argument("--head-hidden-dim", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4, help="Base LR before sqrt(global_batch/1024) scaling")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--weight-decay-end", type=float, default=0.4)
    p.add_argument("--teacher-momentum", type=float, default=0.994)
    p.add_argument("--teacher-temperature", type=float, default=0.07)
    p.add_argument("--teacher-warmup-temperature", type=float, default=0.04)
    p.add_argument("--teacher-warmup-steps", type=int, default=5_000)
    p.add_argument("--mask-min", type=float, default=0.1)
    p.add_argument("--mask-max", type=float, default=0.5)
    p.add_argument("--ibot-weight", type=float, default=1.0)
    p.add_argument("--koleo-weight", type=float, default=0.1)
    p.add_argument("--gram-weight", type=float, default=1.0)
    p.add_argument("--gram-max-tokens", type=int, default=1024)
    p.add_argument("--local-crops", type=int, default=8)
    p.add_argument("--global-shape", type=int, nargs=3, default=None,
                   help="Smoke/debug crop override; production uses the stage default")
    p.add_argument("--local-shape", type=int, nargs=3, default=None,
                   help="Smoke/debug crop override; production uses the stage default")
    p.add_argument("--target-spacing", type=float, nargs=3, default=(1.0, 0.5, 0.5))
    p.add_argument("--target-shape", type=int, nargs=3, default=(256, 384, 384))
    p.add_argument("--posterior-shift-mm", type=float, default=15.0)
    p.add_argument("--cross-sequence-probability", type=float, default=0.75)
    p.add_argument("--candidate-trials", type=int, default=12)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", default="latest", help="latest, none, or a checkpoint path")
    p.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-studies", type=int, default=None, help="Smoke only: cap studies assigned to each rank")
    p.add_argument("--deadline-margin-seconds", type=int, default=900)
    return p.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    else:
        rank = local = 0
        world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def read_train_files(
    preprocessed_dir: str,
    splits_csv: str | None,
    split: str,
    space: str = "atlas_space",
) -> list[str]:
    return [sample["cache_path"] for sample in discover_atlas_cache(
        preprocessed_dir, splits_csv=splits_csv, split=split, space=space
    )]


def cache_fingerprint(
    files: list[str], world: int = 1, max_studies_per_rank: int | None = None
) -> str:
    """Fingerprint ordered study IDs, sequence shapes, and source file sizes."""
    records = []
    for path in sorted(files):
        records.append((Path(path).stem, npz_volume_shape(path), os.path.getsize(path)))
    payload = json.dumps(
        {"records": records, "world": world, "max_studies_per_rank": max_studies_per_rank},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def raw_fingerprint(
    samples: list[dict],
    world: int = 1,
    max_studies_per_rank: int | None = None,
    target_spacing: tuple[float, float, float] = (1.0, 0.5, 0.5),
    target_shape: tuple[int, int, int] = (256, 384, 384),
    posterior_shift_mm: float = 15.0,
) -> str:
    """Fingerprint raw studies without tying exact resume to an absolute mount path."""
    records = []
    for sample in sorted(samples, key=lambda item: item["study_uid"]):
        images = [
            (Path(path).name, os.path.getsize(path))
            for path in sorted(sample["image_paths"])
        ]
        records.append((sample["study_uid"], images))
    payload = json.dumps(
        {
            "records": records,
            "world": world,
            "max_studies_per_rank": max_studies_per_rank,
            "preprocessing": {
                "space": "atlas_space",
                "normalizer": "zscore",
                "target_spacing": list(target_spacing),
                "target_shape": list(target_shape),
                "posterior_shift_mm": posterior_shift_mm,
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def balanced_file_assignment(
    files: list[str],
    world: int,
    weights: dict[str, int] | None = None,
) -> list[list[str]]:
    """Greedy deterministic file assignment by bytes or supplied workload."""
    bins: list[list[str]] = [[] for _ in range(world)]
    primary_loads = [0] * world
    byte_loads = [0] * world
    sized = [
        (int(weights[path]) if weights is not None else os.path.getsize(path), os.path.getsize(path), path)
        for path in files
    ]
    sized.sort(key=lambda x: (-x[0], -x[1], x[2]))
    for primary, size, path in sized:
        rank = min(range(world), key=lambda r: (primary_loads[r], byte_loads[r], r))
        bins[rank].append(path)
        primary_loads[rank] += primary
        byte_loads[rank] += size
    return bins


def balanced_sample_assignment(samples: list[dict], world: int) -> list[list[dict]]:
    """Greedily balance raw studies by sequence count, then compressed bytes."""
    bins: list[list[dict]] = [[] for _ in range(world)]
    sequence_loads = [0] * world
    byte_loads = [0] * world
    sized = []
    for sample in samples:
        size = sum(os.path.getsize(path) for path in sample["image_paths"])
        sized.append((int(sample["n_sequences"]), size, sample["study_uid"], sample))
    sized.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for sequences, size, _, sample in sized:
        rank = min(range(world), key=lambda r: (sequence_loads[r], byte_loads[r], r))
        bins[rank].append(sample)
        sequence_loads[rank] += sequences
        byte_loads[rank] += size
    return bins


def distributed_raw_assignment(
    *,
    data_folder: str,
    splits_csv: str | None,
    split: str,
    output: Path,
    world: int,
    rank: int,
    max_studies_per_rank: int | None,
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    posterior_shift_mm: float,
) -> tuple[list[dict], str]:
    """Discover once on rank 0 and publish one small assignment file per rank."""
    assignment_dir = output / "dataset_assignments"
    metadata_path = assignment_dir / "metadata.json"
    if rank == 0:
        samples = discover_raw_atlas_split(
            data_folder, splits_csv=splits_csv, split=split
        )
        if len(samples) < world:
            raise RuntimeError(f"{len(samples)} atlas studies cannot supply {world} ranks")
        fingerprint = raw_fingerprint(
            samples,
            world,
            max_studies_per_rank,
            target_spacing,
            target_shape,
            posterior_shift_mm,
        )
        assignments = balanced_sample_assignment(samples, world)
        if max_studies_per_rank:
            assignments = [items[:max_studies_per_rank] for items in assignments]
        if any(not items for items in assignments):
            raise RuntimeError("Raw atlas assignment left at least one distributed rank empty")
        assignment_dir.mkdir(parents=True, exist_ok=True)
        for assigned_rank, items in enumerate(assignments):
            destination = assignment_dir / f"rank_{assigned_rank:04d}.json"
            temporary = destination.with_suffix(f".json.partial.{os.getpid()}")
            temporary.write_text(json.dumps(items, separators=(",", ":")))
            os.replace(temporary, destination)
        metadata = {
            "format": "mrrate_atlas_raw_assignment_v1",
            "world_size": world,
            "dataset_fingerprint": fingerprint,
            "split": split,
        }
        temporary = metadata_path.with_suffix(f".json.partial.{os.getpid()}")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        os.replace(temporary, metadata_path)
    if dist.is_initialized():
        dist.barrier()
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "mrrate_atlas_raw_assignment_v1":
        raise RuntimeError(f"Invalid raw atlas assignment metadata: {metadata_path}")
    if int(metadata.get("world_size", -1)) != world:
        raise RuntimeError("Raw atlas assignment world size changed during startup")
    assigned = json.loads((assignment_dir / f"rank_{rank:04d}.json").read_text())
    if not assigned:
        raise RuntimeError(f"Rank {rank} received no raw atlas studies")
    return assigned, str(metadata["dataset_fingerprint"])


def stage_files(files: list[str], root: str, rank: int) -> list[str]:
    if not root:
        return files
    target = Path(root) / f"rank_{rank}"
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for i, source in enumerate(files):
        dest = target / Path(source).name
        if not dest.exists() or dest.stat().st_size != os.path.getsize(source):
            tmp = dest.with_suffix(dest.suffix + ".partial")
            shutil.copyfile(source, tmp)
            os.replace(tmp, dest)
        out.append(str(dest))
        if rank == 0 and (i % 10 == 0 or i + 1 == len(files)):
            print(f"[stage] {i + 1}/{len(files)} local shards", flush=True)
    return out


def build_backbone(arch: str) -> DinoVisionTransformer3D:
    common = dict(
        volume_size=(64, 384, 384),
        patch_size=(2, 16, 16),
        voxel_spacing_mm=(1.0, 0.5, 0.5),
        in_chans=1,
        n_storage_tokens=4,
        drop_path_rate=0.4 if arch == "7b" else (0.3 if arch != "tiny" else 0.0),
        layerscale_init=1e-5,
    )
    if arch == "7b":
        return vit_7b_3d(**common)
    if arch == "hplus":
        return vit_hplus_3d(**common)
    if arch == "large":
        return vit_large_3d(**common)
    return DinoVisionTransformer3D(embed_dim=192, depth=2, num_heads=3, ffn_ratio=2, **common)


def cosine(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return end
    x = min(max(step / (total - 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * x))


def lr_at(args: argparse.Namespace, step: int, scaled_peak: float) -> float:
    if step < args.warmup_steps:
        return scaled_peak * (step + 1) / max(1, args.warmup_steps)
    return cosine(scaled_peak, args.min_lr, step - args.warmup_steps, args.steps - args.warmup_steps)


def teacher_temp_at(args: argparse.Namespace, step: int) -> float:
    if step >= args.teacher_warmup_steps:
        return args.teacher_temperature
    alpha = step / max(1, args.teacher_warmup_steps)
    return args.teacher_warmup_temperature + alpha * (args.teacher_temperature - args.teacher_warmup_temperature)


def unwrap(model: torch.nn.Module) -> DINO3DLearner:
    return model.module if isinstance(model, DDP) else model


def checkpoint_path(output: Path, resume: str) -> Path | None:
    if not resume or resume.lower() == "none":
        return None
    return output / "latest.pt" if resume == "latest" else Path(resume)


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_offset: int,
    args: argparse.Namespace,
    step: int,
    rank: int,
) -> None:
    if dist.is_initialized():
        dist.barrier()
    local_rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if dist.is_initialized():
        rng_by_rank = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    if rank == 0:
        state = {
            "format": "mrrate_atlas_dinov3d_full_v1",
            "step": step,
            "stage": args.stage,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "args": vars(args),
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler_offset": int(sampler_offset),
            "rng_by_rank": rng_by_rank,
            "official_dinov3_commit": "6876159a11b4df116f30f667f8c9888617df0751",
        }
        tmp = output / f"latest.pt.partial.{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, output / "latest.pt")
        numbered = output / f"step_{step:08d}.pt"
        try:
            os.link(output / "latest.pt", numbered)
        except OSError:
            shutil.copyfile(output / "latest.pt", numbered)
        old = sorted(output.glob("step_*.pt"))
        for path in old[: -args.keep_checkpoints]:
            path.unlink(missing_ok=True)
        print(f"[checkpoint] full training state saved at step {step}: {numbered}", flush=True)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: InfiniteStudySampler,
    device: torch.device,
    rank: int,
    current_stage: str,
    dataset_fingerprint: str | None,
    world: int,
) -> int:
    if not path.exists():
        if rank == 0:
            print(f"[resume] {path} not found; starting from scratch", flush=True)
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "mrrate_atlas_dinov3d_full_v1":
        raise RuntimeError(
            f"Checkpoint format {state.get('format')!r} is not atlas-space MR DINO"
        )
    if int(state.get("world_size", world)) != world:
        raise RuntimeError("Exact DDP resume requires the checkpoint's original world size")
    saved_fingerprint = state.get("args", {}).get("dataset_fingerprint")
    if saved_fingerprint and saved_fingerprint != dataset_fingerprint:
        raise RuntimeError("Training cache/split changed since this checkpoint")
    missing, unexpected = unwrap(model).load_state_dict(state["model"], strict=False)
    # A pretrain checkpoint intentionally lacks the new frozen Gram anchor.
    meaningful_missing = [k for k in missing if not k.startswith("gram_anchor.")]
    if meaningful_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={meaningful_missing[:8]} unexpected={unexpected[:8]}")
    if unwrap(model).gram_anchor is not None and any(k.startswith("gram_anchor.") for k in missing):
        unwrap(model).reset_gram_anchor_from_teacher()
    optimizer.load_state_dict(state["optimizer"])
    source_stage = str(state.get("stage", "pretrain"))
    step = int(state["step"]) if source_stage == current_stage else 0
    sampler.set_offset(int(state.get("sampler_offset", step)))
    if "rng_by_rank" in state and rank < len(state["rng_by_rank"]):
        rng = state["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"], device)
    else:
        # Backward compatibility with early smoke checkpoints.
        seed = int(state["args"].get("seed", 3407)) + rank + 1_000_003 * step
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    if rank == 0:
        transition = "" if source_stage == current_stage else f"; stage transition {source_stage}->{current_stage}, schedule reset"
        print(f"[resume] loaded {path} at completed step {step}{transition}", flush=True)
    del state
    return step


class StopController:
    def __init__(self, deadline: float | None) -> None:
        self.requested = False
        self.deadline = deadline
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._signal)

    def _signal(self, signum, frame) -> None:
        del signum, frame
        self.requested = True

    def should_stop(self, device: torch.device) -> bool:
        local = self.requested or (self.deadline is not None and time.time() >= self.deadline)
        flag = torch.tensor(int(local), device=device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def main() -> int:
    args = parse_args()
    rank, world, local_rank, device = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)  # identical student initialization before DDP
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    if dist.is_initialized():
        dist.barrier()

    if args.preprocessed_dir:
        files = read_train_files(args.preprocessed_dir, args.splits_csv, args.split, args.space)
        if len(files) < world:
            raise RuntimeError(f"{len(files)} cache studies cannot supply {world} distributed ranks")
        args.dataset_fingerprint = cache_fingerprint(files, world, args.max_studies)
        sequence_counts = {path: npz_volume_shape(path)[0] for path in files}
        assigned = balanced_file_assignment(files, world, weights=sequence_counts)[rank]
        if args.max_studies:
            assigned = assigned[: args.max_studies]
        assigned = stage_files(assigned, os.environ.get("MRDINO_LOCAL_DATA_DIR", ""), rank)
        dataset_kwargs = {
            "preprocessed_dir": args.preprocessed_dir,
            "cache_files": assigned,
        }
    else:
        assigned_samples, args.dataset_fingerprint = distributed_raw_assignment(
            data_folder=args.data_folder,
            splits_csv=args.splits_csv,
            split=args.split,
            output=output,
            world=world,
            rank=rank,
            max_studies_per_rank=args.max_studies,
            target_spacing=tuple(args.target_spacing),
            target_shape=tuple(args.target_shape),
            posterior_shift_mm=args.posterior_shift_mm,
        )
        dataset_kwargs = {
            "data_folder": args.data_folder,
            "raw_samples": assigned_samples,
        }
    if rank == 0:
        with open(output / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    base_crop_spec = stage_crop_spec(args.stage)
    crop_spec = type(base_crop_spec)(
        global_shape=tuple(args.global_shape) if args.global_shape else base_crop_spec.global_shape,
        local_shape=tuple(args.local_shape) if args.local_shape else base_crop_spec.local_shape,
        local_crops=args.local_crops,
    )
    dataset = MRAtlasDINO3DDataset(
        **dataset_kwargs,
        splits_csv=args.splits_csv,
        split=args.split,
        space=args.space,
        crop_spec=crop_spec,
        target_spacing=tuple(args.target_spacing),
        target_shape=tuple(args.target_shape),
        posterior_shift_mm=args.posterior_shift_mm,
        cross_sequence_probability=args.cross_sequence_probability,
        candidate_trials=args.candidate_trials,
        seed=args.seed,
    )
    local_instances = torch.tensor(len(dataset), device=device, dtype=torch.long)
    total_instances = local_instances.clone()
    local_studies = torch.tensor(dataset.n_studies, device=device, dtype=torch.long)
    total_studies = local_studies.clone()
    if dist.is_initialized():
        dist.all_reduce(total_instances)
        dist.all_reduce(total_studies)
    sampler = InfiniteStudySampler(
        len(dataset),
        seed=args.seed + rank * 7919,
        group_sizes=[int(sample["n_sequences"]) for sample in dataset.samples],
    )
    collate = functools.partial(
        collate_dino3d,
        patch_size=(2, 16, 16),
        mask_ratio=(args.mask_min, args.mask_max),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers > 0 else None,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate,
    )

    backbone = build_backbone(args.arch)
    learner = DINO3DLearner(
        backbone,
        prototypes=args.prototypes,
        head_hidden_dim=args.head_hidden_dim,
        loss_weights=LossWeights(
            dino=1.0,
            ibot=args.ibot_weight,
            koleo=args.koleo_weight,
            gram=args.gram_weight if args.stage in {"gram", "highres"} else 0.0,
        ),
        gram_max_tokens=args.gram_max_tokens,
        with_gram_anchor=args.stage in {"gram", "highres"},
    )
    if args.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        for i, block in enumerate(learner.student.backbone.blocks):
            learner.student.backbone.blocks[i] = checkpoint_wrapper(block)
    learner.to(device)
    trainable = [p for p in learner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if world > 1:
        learner = DDP(
            learner,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100,
        )

    start = load_checkpoint(
        checkpoint_path(output, args.resume), learner, optimizer, sampler, device, rank,
        args.stage, args.dataset_fingerprint, world,
    ) if checkpoint_path(output, args.resume) else 0
    consumed_samples = sampler.offset
    # Cross-stage resume fixes the new Gram target to the incoming EMA teacher.
    if args.stage in {"gram", "highres"} and start == 0 and unwrap(learner).gram_anchor is not None:
        unwrap(learner).reset_gram_anchor_from_teacher()

    global_batch = world * args.batch_size
    study_balance_scale = float(total_instances) / float(total_studies)
    scaled_peak_lr = args.lr * 4.0 * math.sqrt(global_batch / 1024.0)
    n_params = sum(p.numel() for p in unwrap(learner).student_module.backbone.parameters())
    if rank == 0:
        print(
            f"[mrdino3d] stage={args.stage} arch={args.arch} backbone={n_params/1e6:.1f}M "
            f"world={world} global_batch={global_batch} studies={int(total_studies)} "
            f"sequences={int(total_instances)} "
            f"global_crop={crop_spec.global_shape} tokens={math.prod(v//p for v,p in zip(crop_spec.global_shape,(2,16,16)))} "
            f"peak_lr={scaled_peak_lr:.3g}",
            flush=True,
        )

    deadline_epoch = float(os.environ.get("MRDINO_DEADLINE_EPOCH", "0") or 0)
    deadline = deadline_epoch if deadline_epoch else None
    stopper = StopController(deadline)
    iterator = iter(loader)
    log_path = output / "metrics.jsonl"
    learner.train()
    last_time = time.time()
    for step in range(start, args.steps):
        if stopper.should_stop(device):
            save_checkpoint(output, learner, optimizer, consumed_samples, args, step, rank)
            if rank == 0:
                (output / "REQUEUE_REQUESTED").touch()
            if dist.is_initialized():
                dist.destroy_process_group()
            return 75
        batch = next(iterator)
        consumed_samples += args.batch_size
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        if device.type == "cpu":
            for key in ("teacher_global", "student_global", "student_local"):
                batch[key] = batch[key].float()
        batch["loss_weights"] = batch["sample_weights"] * study_balance_scale
        lr = lr_at(args, step, scaled_peak_lr)
        wd = cosine(args.weight_decay, args.weight_decay_end, step, args.steps)
        momentum = cosine(args.teacher_momentum, 1.0, step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = learner(batch, teacher_temperature=teacher_temp_at(args, step), step=step)
            backward_loss = loss
        finite = torch.isfinite(backward_loss.detach()).to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"Non-finite distributed loss at step {step}: {loss.detach()}")
        backward_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 3.0)
        optimizer.step()
        unwrap(learner).update_teacher(momentum)

        if (step + 1) % args.log_every == 0:
            values = torch.stack([metrics[k].float() for k in ("loss", "dino", "ibot", "koleo", "gram")])
            if dist.is_initialized():
                dist.all_reduce(values)
                values /= world
            now = time.time()
            elapsed = now - last_time
            last_time = now
            if rank == 0:
                record = {
                    "step": step + 1,
                    "stage": args.stage,
                    "sequence_epoch": (
                        (step + 1) * global_batch / int(total_instances)
                    ),
                    "loss": float(values[0]),
                    "dino": float(values[1]),
                    "ibot": float(values[2]),
                    "koleo": float(values[3]),
                    "gram": float(values[4]),
                    "lr": lr,
                    "weight_decay": wd,
                    "teacher_momentum": momentum,
                    "grad_norm": float(grad_norm),
                    "steps_per_second": args.log_every / elapsed,
                    "peak_gpu_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                    ),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)

        if (step + 1) % args.save_every == 0:
            save_checkpoint(output, learner, optimizer, consumed_samples, args, step + 1, rank)

    if args.steps % args.save_every:
        save_checkpoint(output, learner, optimizer, consumed_samples, args, args.steps, rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
