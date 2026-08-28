"""FSDP2 training for the literal DINOv3 ViT-7B on coregistered MR-RATE."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader

from dinov3.checkpointer import load_checkpoint as dcp_load
from dinov3.checkpointer import save_checkpoint as dcp_save
from dinov3.layers.fp8_linear import convert_linears_to_fp8

from .data import MRCoregDINO3DDataset, InfiniteStudySampler, collate_dino3d, stage_crop_spec
from .fp8 import enable_fsdp_mixed_precision_fp8
from .objective import DINO3DLearner, LossWeights
from .train_ddp import (
    StopController,
    balanced_file_assignment,
    build_backbone,
    cache_fingerprint,
    cosine,
    read_train_files,
    setup_distributed,
    stage_files,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocessed-dir", required=True,
                   help="MR-RATE preprocessing cache root containing coreg_space/*.npz")
    p.add_argument("--splits-csv")
    p.add_argument("--split", default="train")
    p.add_argument("--space", default="coreg_space", choices=("coreg_space",))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=("pretrain", "gram", "highres"), default="pretrain")
    p.add_argument("--arch", choices=("tiny", "7b"), default="7b")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Fixed study microbatches per optimizer update on every rank",
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--local-crops", type=int, default=8)
    p.add_argument("--global-shape", type=int, nargs=3, default=None,
                   help="Smoke/debug crop override; production uses the stage default")
    p.add_argument("--local-shape", type=int, nargs=3, default=None,
                   help="Smoke/debug crop override; production uses the stage default")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--dino-prototypes", type=int, default=262_144)
    p.add_argument("--ibot-prototypes", type=int, default=98_304)
    p.add_argument("--dino-head-hidden-dim", type=int, default=8192)
    p.add_argument("--ibot-head-hidden-dim", type=int, default=4096)
    p.add_argument("--dino-bottleneck-dim", type=int, default=512)
    p.add_argument("--ibot-bottleneck-dim", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-5, help="Peak at global batch 1024 before sqrt scaling")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--weight-decay-end", type=float, default=0.04)
    p.add_argument("--teacher-momentum", type=float, default=0.994)
    p.add_argument("--teacher-temperature", type=float, default=0.07)
    p.add_argument("--teacher-warmup-temperature", type=float, default=0.04)
    p.add_argument("--teacher-warmup-steps", type=int, default=10_000)
    p.add_argument("--mask-min", type=float, default=0.1)
    p.add_argument("--mask-max", type=float, default=0.5)
    p.add_argument("--ibot-weight", type=float, default=1.0)
    p.add_argument("--koleo-weight", type=float, default=0.1)
    p.add_argument("--gram-weight", type=float, default=1.0)
    p.add_argument("--gram-max-tokens", type=int, default=1024)
    p.add_argument("--ibot-loss-chunk-size", type=int, default=1024)
    p.add_argument("--cross-sequence-probability", type=float, default=0.75)
    p.add_argument("--candidate-trials", type=int, default=12)
    p.add_argument("--clip-grad", type=float, default=30.0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", default="latest", help="latest, none, or a DCP checkpoint directory")
    p.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--selective-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-studies", type=int, default=None, help="Smoke only: cap studies per rank")
    args = p.parse_args()
    # A running Slurm allocation can cleanly restart the worker step without
    # relinquishing its nodes.  Keep schedule changes in a small, auditable
    # control file next to the checkpoints so every rank and every later
    # requeue observes the same values.  Existing workers never reread this
    # file, making creation safe before an allocation-local worker handoff.
    control_path = Path(args.output_dir) / "schedule_override.json"
    args.schedule_override = None
    if control_path.is_file():
        control = json.loads(control_path.read_text())
        if control.get("format") != "mrdino3d_schedule_override_v1":
            raise ValueError(f"Unexpected schedule override format: {control_path}")
        allowed = {
            "warmup_steps",
            "teacher_warmup_steps",
            "warmup_transition_step",
            "warmup_transition_lr_factor",
            "teacher_warmup_transition_temperature",
        }
        unknown = set(control) - allowed - {"format", "created_from_checkpoint"}
        if unknown:
            raise ValueError(f"Unknown schedule override keys: {sorted(unknown)}")
        for key in allowed:
            if key in control:
                setattr(args, key, control[key])
        args.schedule_override = str(control_path.resolve())
    return args


def warmup_lr_factor(args: argparse.Namespace, step: int) -> float:
    """Continuous LR ramp, optionally anchored at a hot-restart checkpoint."""
    end = int(args.warmup_steps)
    start = int(getattr(args, "warmup_transition_step", 0))
    start_factor = float(getattr(args, "warmup_transition_lr_factor", 0.0))
    if end < 1 or not 0 <= start < end:
        raise ValueError("warmup transition must satisfy 0 <= start < warmup_steps")
    if not 0.0 <= start_factor <= 1.0:
        raise ValueError("warmup_transition_lr_factor must be in [0, 1]")
    if step >= end:
        return 1.0
    # ``start`` is the number of optimizer steps already completed in the
    # checkpoint.  The next loop index is therefore ``start`` and advances by
    # one increment, matching the original (step + 1) / warmup_steps schedule.
    alpha = min(1.0, max(0.0, (step + 1 - start) / max(1, end - start)))
    return start_factor + alpha * (1.0 - start_factor)


def teacher_temperature_at(args: argparse.Namespace, step: int) -> float:
    """Continuous teacher-temperature ramp for an allocation-local restart."""
    end = int(args.teacher_warmup_steps)
    start = int(getattr(args, "warmup_transition_step", 0))
    start_temperature = float(
        getattr(
            args,
            "teacher_warmup_transition_temperature",
            args.teacher_warmup_temperature,
        )
    )
    if end < 1 or not 0 <= start < end:
        raise ValueError(
            "teacher warmup transition must satisfy 0 <= start < teacher_warmup_steps"
        )
    if start_temperature <= 0 or args.teacher_temperature <= 0:
        raise ValueError("teacher temperatures must be positive")
    if step >= end:
        return float(args.teacher_temperature)
    alpha = min(1.0, max(0.0, (step - start) / max(1, end - start)))
    return start_temperature + alpha * (
        float(args.teacher_temperature) - start_temperature
    )


def _checkpoint_wrapper(selective: bool):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    if not selective:
        return checkpoint_wrapper
    from torch.utils.checkpoint import create_selective_checkpoint_contexts
    save_ops = [
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    ]
    return functools.partial(
        checkpoint_wrapper,
        context_fn=functools.partial(create_selective_checkpoint_contexts, save_ops),
        preserve_rng_state=True,
    )


def _parallelize_network(network, mesh, mp_policy) -> None:
    cfg = {"mesh": mesh, "mp_policy": mp_policy}
    blocks = network.backbone.blocks
    for i, block in enumerate(blocks):
        blocks[i] = fully_shard(block, **cfg, reshard_after_forward=True)
    for previous, following in zip(blocks, blocks[1:]):
        previous.set_modules_to_forward_prefetch([following])
        following.set_modules_to_backward_prefetch([previous])
    fully_shard(network.backbone, **cfg, reshard_after_forward=True)
    register_fsdp_forward_method(network.backbone, "get_intermediate_layers")
    network.dino_head = fully_shard(network.dino_head, **cfg, reshard_after_forward=True)
    network.ibot_head = fully_shard(network.ibot_head, **cfg, reshard_after_forward=True)


def build_learner(args: argparse.Namespace, device: torch.device) -> DINO3DLearner:
    if args.fp8:
        enable_fsdp_mixed_precision_fp8()
    with torch.device("meta"):
        backbone = build_backbone(args.arch)
        if args.fp8:
            backbone = convert_linears_to_fp8(backbone, filter="blocks")
        learner = DINO3DLearner(
            backbone,
            dino_prototypes=args.dino_prototypes,
            ibot_prototypes=args.ibot_prototypes,
            dino_head_hidden_dim=args.dino_head_hidden_dim,
            ibot_head_hidden_dim=args.ibot_head_hidden_dim,
            dino_bottleneck_dim=args.dino_bottleneck_dim,
            ibot_bottleneck_dim=args.ibot_bottleneck_dim,
            loss_weights=LossWeights(
                dino=1.0,
                ibot=args.ibot_weight,
                koleo=args.koleo_weight,
                gram=args.gram_weight if args.stage in {"gram", "highres"} else 0.0,
            ),
            gram_max_tokens=args.gram_max_tokens,
            ibot_loss_chunk_size=args.ibot_loss_chunk_size,
            with_gram_anchor=args.stage in {"gram", "highres"},
            initialize_weights=False,
        )
    if args.activation_checkpointing:
        wrapper = _checkpoint_wrapper(args.selective_checkpointing)
        for i, block in enumerate(learner.student.backbone.blocks):
            learner.student.backbone.blocks[i] = wrapper(block)
    if args.compile:
        for network in (learner.student, learner.teacher, learner.gram_anchor):
            if network is None:
                continue
            for block in network.backbone.blocks:
                block.compile()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    _parallelize_network(learner.student, mesh, mp_policy)
    _parallelize_network(learner.teacher, mesh, mp_policy)
    if learner.gram_anchor is not None:
        _parallelize_network(learner.gram_anchor, mesh, mp_policy)
    learner.to_empty(device=device)
    learner.initialize_student_teacher()
    return learner


def checkpoint_root(output: Path) -> Path:
    return output / "checkpoints"


def resolve_checkpoint(output: Path, resume: str) -> Path | None:
    if not resume or resume.lower() == "none":
        return None
    if resume != "latest":
        return Path(resume)
    marker = checkpoint_root(output) / "latest.txt"
    if not marker.exists():
        return None
    path = checkpoint_root(output) / marker.read_text().strip()
    return path if (path / "COMPLETE").exists() else None


def runtime_state(sampler_offset: int) -> dict:
    return {
        "sampler_offset": int(sampler_offset),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def save_checkpoint(
    output: Path,
    learner: DINO3DLearner,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    sampler_offset: int,
    rank: int,
) -> None:
    root = checkpoint_root(output)
    path = root / f"step_{step:08d}"
    dcp_save(path, iteration=step, model=learner, optimizer=optimizer, overwrite=True)
    tmp = path / f"runtime_rank{rank:04d}.pt.partial"
    torch.save(runtime_state(sampler_offset), tmp)
    os.replace(tmp, path / f"runtime_rank{rank:04d}.pt")
    if rank == 0:
        (path / "metadata.json").write_text(json.dumps({
            "format": "mrrate_coreg_dinov3d_fsdp2_v1",
            "step": step,
            "stage": args.stage,
            "sampler_offset_per_rank": sampler_offset,
            "world_size": dist.get_world_size(),
            "args": vars(args),
            "official_dinov3_commit": "6876159a11b4df116f30f667f8c9888617df0751",
        }, indent=2, sort_keys=True))
    dist.barrier()
    if rank == 0:
        (path / "COMPLETE").touch()
        latest_tmp = root / "latest.txt.partial"
        latest_tmp.write_text(path.name + "\n")
        os.replace(latest_tmp, root / "latest.txt")
        old = sorted(p for p in root.glob("step_*") if p.is_dir() and (p / "COMPLETE").exists())
        for stale in old[:-args.keep_checkpoints]:
            shutil.rmtree(stale)
        print(f"[checkpoint] distributed full state saved at step {step}: {path}", flush=True)
    dist.barrier()


def load_checkpoint(
    path: Path | None,
    learner: DINO3DLearner,
    optimizer: torch.optim.Optimizer,
    sampler: InfiniteStudySampler,
    args: argparse.Namespace,
    rank: int,
) -> int:
    if path is None:
        if rank == 0:
            print("[resume] no complete checkpoint; starting from scratch", flush=True)
        return 0
    if not (path / "COMPLETE").exists():
        raise RuntimeError(f"Incomplete checkpoint: {path}")
    metadata = json.loads((path / "metadata.json").read_text())
    saved_world = int(metadata.get("world_size", dist.get_world_size()))
    if saved_world != dist.get_world_size():
        raise RuntimeError(
            f"Checkpoint world size {saved_world} != current {dist.get_world_size()}; "
            "exact per-rank sampler/RNG resume requires the same world size"
        )
    saved_fingerprint = metadata.get("args", {}).get("dataset_fingerprint")
    if saved_fingerprint and saved_fingerprint != args.dataset_fingerprint:
        raise RuntimeError(
            "Coregistered training cache/split changed since the checkpoint; "
            "refusing an inexact sampler resume"
        )
    source_stage = metadata["stage"]
    loaded_step = int(dcp_load(
        path,
        model=learner,
        optimizer=optimizer,
        strict_loading=(source_stage == args.stage),
    ))
    state = torch.load(path / f"runtime_rank{rank:04d}.pt", map_location="cpu", weights_only=False)
    sampler.set_offset(int(state["sampler_offset"]))
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.cuda.set_rng_state(state["cuda"])
    step = loaded_step if source_stage == args.stage else 0
    if source_stage != args.stage and learner.gram_anchor is not None:
        learner.reset_gram_anchor_from_teacher()
    if rank == 0:
        transition = "" if source_stage == args.stage else f"; stage transition {source_stage}->{args.stage}, schedule reset"
        print(f"[resume] loaded {path} at completed step {step}{transition}", flush=True)
    return step


def scalar_norm(value: torch.Tensor) -> float:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return float(value)


def main() -> int:
    args = parse_args()
    rank, world, _, device = setup_distributed()
    if world < 2:
        raise RuntimeError("The 7B path requires distributed FSDP2 (WORLD_SIZE >= 2)")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + rank)

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_root(output).mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    dist.barrier()

    files = read_train_files(args.preprocessed_dir, args.splits_csv, args.split)
    if len(files) < world:
        raise RuntimeError(f"{len(files)} cache studies cannot supply {world} distributed ranks")
    args.dataset_fingerprint = cache_fingerprint(files, world, args.max_studies)
    if rank == 0:
        (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    from .data import npz_volume_shape
    sequence_counts = {path: npz_volume_shape(path)[0] for path in files}
    assigned = balanced_file_assignment(files, world, weights=sequence_counts)[rank]
    if args.max_studies:
        assigned = assigned[: args.max_studies]
    assigned = stage_files(assigned, os.environ.get("MRDINO_LOCAL_DATA_DIR", ""), rank)
    base_crop_spec = stage_crop_spec(args.stage)
    crop_spec = type(base_crop_spec)(
        global_shape=tuple(args.global_shape) if args.global_shape else base_crop_spec.global_shape,
        local_shape=tuple(args.local_shape) if args.local_shape else base_crop_spec.local_shape,
        local_crops=args.local_crops,
    )
    dataset = MRCoregDINO3DDataset(
        preprocessed_dir=args.preprocessed_dir,
        cache_files=assigned,
        splits_csv=args.splits_csv,
        split=args.split,
        space=args.space,
        crop_spec=crop_spec,
        cross_sequence_probability=args.cross_sequence_probability,
        candidate_trials=args.candidate_trials,
        seed=args.seed,
    )
    local_instances = torch.tensor(len(dataset), device=device, dtype=torch.long)
    total_instances = local_instances.clone()
    dist.all_reduce(total_instances)
    local_studies = torch.tensor(dataset.n_studies, device=device, dtype=torch.long)
    total_studies = local_studies.clone()
    dist.all_reduce(total_studies)
    sampler = InfiniteStudySampler(
        len(dataset),
        seed=args.seed + rank * 7919,
        group_sizes=[int(sample["n_sequences"]) for sample in dataset.samples],
    )

    learner = build_learner(args, device)
    trainable = list(learner.student.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
        fused=True,
    )
    start = load_checkpoint(resolve_checkpoint(output, args.resume), learner, optimizer, sampler, args, rank)
    transition_step = int(getattr(args, "warmup_transition_step", 0))
    if start < transition_step:
        raise ValueError(
            f"Checkpoint step {start} precedes schedule transition step {transition_step}"
        )
    consumed_samples = sampler.offset

    collate = functools.partial(
        collate_dino3d,
        patch_size=(2, 16, 16),
        mask_ratio=(args.mask_min, args.mask_max),
    )
    loader_generator = torch.Generator().manual_seed(args.seed + rank * 104729)
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
        generator=loader_generator,
    )
    iterator = iter(loader)

    sinkhorn_batch = world * args.batch_size
    effective_global_batch = sinkhorn_batch * args.grad_accum_steps
    peak_lr = args.lr * math.sqrt(effective_global_batch / 1024.0)
    study_balance_scale = float(total_instances) / float(total_studies)
    n_params = sum(p.numel() for p in learner.student.backbone.parameters())
    if rank == 0:
        print(
            f"[mrdino3d] FSDP2 stage={args.stage} arch={args.arch} backbone={n_params/1e9:.3f}B "
            f"world={world} sinkhorn_batch={sinkhorn_batch} "
            f"effective_global_batch={effective_global_batch} studies={int(total_studies)} "
            f"sequences={int(total_instances)} "
            f"global_crop={crop_spec.global_shape} "
            f"tokens={math.prod(v//p for v,p in zip(crop_spec.global_shape,(2,16,16)))} "
            f"peak_lr={peak_lr:.3g} fp8={args.fp8}",
            flush=True,
        )
        print(
            "[schedule] "
            f"warmup_steps={args.warmup_steps} "
            f"teacher_warmup_steps={args.teacher_warmup_steps} "
            f"transition_step={transition_step} "
            f"transition_lr_factor={float(getattr(args, 'warmup_transition_lr_factor', 0.0)):.8f} "
            f"transition_teacher_temperature="
            f"{float(getattr(args, 'teacher_warmup_transition_temperature', args.teacher_warmup_temperature)):.8f} "
            f"override={args.schedule_override or 'none'}",
            flush=True,
        )

    deadline_epoch = float(os.environ.get("MRDINO_DEADLINE_EPOCH", "0") or 0)
    stopper = StopController(deadline_epoch if deadline_epoch else None)
    log_path = output / "metrics.jsonl"
    learner.train()
    last_time = time.time()

    for step in range(start, args.steps):
        if stopper.should_stop(device):
            save_checkpoint(output, learner, optimizer, args, step, consumed_samples, rank)
            if rank == 0:
                (output / "REQUEUE_REQUESTED").touch()
            dist.destroy_process_group()
            return 75

        if step < args.warmup_steps:
            lr = peak_lr * warmup_lr_factor(args, step)
        else:
            lr = cosine(peak_lr, args.min_lr, step - args.warmup_steps, args.steps - args.warmup_steps)
        wd = cosine(args.weight_decay, args.weight_decay_end, step, args.steps)
        momentum = cosine(args.teacher_momentum, 1.0, step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd

        optimizer.zero_grad(set_to_none=True)
        metric_accumulator = torch.zeros(5, device=device, dtype=torch.float32)
        for microstep in range(args.grad_accum_steps):
            batch = next(iterator)
            consumed_samples += args.batch_size
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            batch["loss_weights"] = batch["sample_weights"] * study_balance_scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = learner(
                    batch,
                    teacher_temperature=teacher_temperature_at(args, step),
                    step=step * args.grad_accum_steps + microstep,
                )
                # Per-sample 1/R importance weighting is already applied by
                # the learner before each SSL loss is reduced.
                backward_loss = loss / args.grad_accum_steps
            finite = torch.isfinite(backward_loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                metric_debug = {
                    key: float(value.detach().float())
                    for key, value in metrics.items()
                }
                raise FloatingPointError(
                    f"Non-finite distributed loss at step {step}, microstep {microstep}: "
                    f"{backward_loss.detach()}; metrics={metric_debug}"
                )
            backward_loss.backward()
            metric_accumulator += torch.stack(
                [metrics[k].float() for k in ("loss", "dino", "ibot", "koleo", "gram")]
            )
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(module.parameters(), args.clip_grad)
            for module in (learner.student.backbone, learner.student.dino_head, learner.student.ibot_head)
        ]
        optimizer.step()
        learner.update_teacher(momentum)

        if (step + 1) % args.log_every == 0:
            values = metric_accumulator / args.grad_accum_steps
            dist.all_reduce(values)
            values /= world
            now = time.time()
            if rank == 0:
                record = {
                    "step": step + 1,
                    "stage": args.stage,
                    "sequence_epoch": (
                        (step + 1)
                        * effective_global_batch
                        / int(total_instances)
                    ),
                    "grad_accum_steps": args.grad_accum_steps,
                    "loss": float(values[0]),
                    "dino": float(values[1]),
                    "ibot": float(values[2]),
                    "koleo": float(values[3]),
                    "gram": float(values[4]),
                    "lr": lr,
                    "weight_decay": wd,
                    "teacher_momentum": momentum,
                    "grad_norm_max": max(scalar_norm(x) for x in grad_norms),
                    "steps_per_second": args.log_every / (now - last_time),
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            last_time = now

        if (step + 1) % args.save_every == 0:
            save_checkpoint(output, learner, optimizer, args, step + 1, consumed_samples, rank)

    if args.steps % args.save_every:
        save_checkpoint(output, learner, optimizer, args, args.steps, consumed_samples, rank)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
