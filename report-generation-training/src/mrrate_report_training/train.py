from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .cache import ExactRaggedTokenDataset
from .config import load_config, require_training_policy
from .mil import infer_mil, load_frozen_mil
from .model import (
    ReportWriter,
    build_gemma_writer,
    label_semantic_embeddings,
    trainable_state_dict,
)
from .online import OnlineSource, verify_frozen_encoder
from .provenance import verify_mil_encoder_provenance
from .targets import ReportTarget, load_target_index


DUMMY_TARGET = ReportTarget("__padding__", "<NONE>")
_CHECKPOINT_AND_STOP = False


def _request_checkpoint(_signal_number, _frame) -> None:
    global _CHECKPOINT_AND_STOP
    _CHECKPOINT_AND_STOP = True


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("MR-RATE report training requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def exact_rank_indices(
    length: int, epoch: int, seed: int, world: int, rank: int, shuffle: bool
) -> list[int]:
    """Every real index exactly once; only explicit -1 no-op slots are padded."""

    if length <= 0 or world <= 0 or not 0 <= rank < world:
        raise ValueError("invalid exact-shard arguments")
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        indices = torch.randperm(length, generator=generator).tolist()
    else:
        indices = list(range(length))
    padded_length = math.ceil(length / world) * world
    indices.extend([-1] * (padded_length - length))
    return indices[rank::world]


def cosine_schedule(
    optimizer: torch.optim.Optimizer, total_updates: int, warmup_ratio: float
):
    warmup = int(total_updates * float(warmup_ratio))

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    label_names: list[str],
    *,
    epoch: int,
    next_slot: int,
    update: int,
    rank: int,
    world: int,
) -> None:
    local_rng = rng_state()
    if world > 1:
        all_rng: list[dict | None] = [None] * world
        dist.all_gather_object(all_rng, local_rng)
    else:
        all_rng = [local_rng]
    if rank:
        return
    bare = model.module if isinstance(model, DistributedDataParallel) else model
    package = {
        "format_version": 1,
        "trainable_state_dict": trainable_state_dict(bare),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": {key: value for key, value in config.items() if key != "_config_path"},
        "label_names": label_names,
        "epoch": epoch,
        "next_slot": next_slot,
        "update": update,
        "rng_by_rank": all_rng,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    labels: list[str],
    rank: int,
) -> tuple[int, int, int]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("label_names") != labels:
        raise ValueError("Resume checkpoint MIL label schema differs")
    bare = model.module if isinstance(model, DistributedDataParallel) else model
    incompatible = bare.load_state_dict(package["trainable_state_dict"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected resume tensors: {unexpected[:5]}")
    optimizer.load_state_dict(package["optimizer"])
    scheduler.load_state_dict(package["scheduler"])
    states = package["rng_by_rank"]
    restore_rng(states[rank] if rank < len(states) else states[0])
    return int(package["epoch"]), int(package["next_slot"]), int(package["update"])


def main() -> None:
    signal.signal(signal.SIGUSR1, _request_checkpoint)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("online", "cached"), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-studies", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--llm-path")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--mil-checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    config["mode"] = args.mode
    if args.llm_path:
        config["llm_path"] = args.llm_path
    if args.encoder_checkpoint:
        config["encoder_checkpoint"] = args.encoder_checkpoint
    if args.mil_checkpoint:
        config["mil_checkpoint"] = args.mil_checkpoint
    require_training_policy(config)
    rank, world, _, device = distributed_setup()
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    targets = load_target_index(config["data"]["reports_csv"])
    mil_head, label_names, thresholds = load_frozen_mil(
        config["mil_checkpoint"],
        config["upstream_root"],
        expected_dim=int(config["encoder"]["dim_latent"]),
    )
    mil_head.to(device).eval()
    thresholds = thresholds.to(device)

    online_source = None
    if args.mode == "cached":
        source = ExactRaggedTokenDataset(
            config["data"]["cached_tokens_dir"],
            "train",
            targets,
            expected_dim=int(config["encoder"]["dim_latent"]),
            expected_label_names=label_names,
        )
        subject_ids = source.subject_ids
    else:
        online_source = OnlineSource(config, device)
        source = online_source
        subject_ids = online_source.subject_ids
        missing = [value for value in subject_ids if value not in targets]
        if missing:
            raise ValueError(
                f"{len(missing)} online studies lack report targets; first={missing[:5]}"
            )
    cache_metadata = source.metadata if args.mode == "cached" else None
    if rank == 0:
        try:
            provenance_result = verify_mil_encoder_provenance(
                config["mil_checkpoint"],
                config["encoder_checkpoint"],
                config["encoder"],
                cache_metadata=cache_metadata,
            )
            provenance_message = {"ok": True, "result": provenance_result}
        except Exception as error:
            provenance_message = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        provenance_message = None
    if world > 1:
        messages = [provenance_message]
        dist.broadcast_object_list(messages, src=0)
        provenance_message = messages[0]
    if not provenance_message["ok"]:
        raise RuntimeError(
            f"MIL/encoder provenance verification failed: "
            f"{provenance_message['error']}"
        )
    if rank == 0:
        print(
            "[provenance] " + json.dumps(provenance_message["result"]),
            flush=True,
        )
    source_length = len(source)
    if args.max_studies:
        source_length = min(source_length, int(args.max_studies))

    writer_config = config["writer"]
    llm, tokenizer, _ = build_gemma_writer(
        config["llm_path"],
        device,
        lora_r=int(writer_config["lora_r"]),
        lora_alpha=int(writer_config["lora_alpha"]),
    )
    semantics = label_semantic_embeddings(llm, tokenizer, label_names)
    model = ReportWriter(
        llm,
        tokenizer,
        semantics,
        visual_dim=int(config["encoder"]["dim_latent"]),
        num_visual_queries=int(writer_config["num_visual_queries"]),
        resampler_depth=int(writer_config["resampler_depth"]),
        resampler_heads=int(writer_config["resampler_heads"]),
        max_target_tokens=int(writer_config["max_target_tokens"]),
    ).to(device)
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"]["gradient_accumulation"])
    slots_per_epoch = math.ceil(source_length / world)
    updates_per_epoch = math.ceil(slots_per_epoch / accumulation)
    scheduler = cosine_schedule(
        optimizer,
        epochs * updates_per_epoch,
        float(config["training"]["warmup_ratio"]),
    )
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False
        )

    start_epoch = start_slot = update = 0
    if args.resume:
        start_epoch, start_slot, update = load_checkpoint(
            args.resume, model, optimizer, scheduler, label_names, rank
        )
    output_dir = Path(config["output_dir"]).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "studies": source_length,
                    "world_size": world,
                    "slots_per_rank": slots_per_epoch,
                    "epochs": epochs,
                    "visual_queries": writer_config["num_visual_queries"],
                    "mil_classes": len(label_names),
                    "localization": False,
                    "replacement_sampling": False,
                }
            ),
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, epochs):
        local_indices = exact_rank_indices(
            source_length,
            epoch,
            seed,
            world,
            rank,
            bool(config["training"]["shuffle"]),
        )
        first_slot = start_slot if epoch == start_epoch else 0
        for slot in range(first_slot, len(local_indices)):
            index = local_indices[slot]
            is_real = index >= 0
            if is_real:
                if args.mode == "cached":
                    item = source[index]
                    tokens = item["tokens"].to(
                        device=device, dtype=torch.bfloat16, non_blocking=True
                    )
                    subject_id = item["subject_id"]
                else:
                    item = online_source.get(index)
                    tokens = item["tokens"]
                    subject_id = item["subject_id"]
                target = targets[subject_id]
            else:
                tokens = torch.zeros(
                    1,
                    int(config["encoder"]["dim_latent"]),
                    dtype=torch.bfloat16,
                    device=device,
                )
                target = DUMMY_TARGET
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mil_logits = infer_mil(mil_head, tokens)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(
                    tokens,
                    mil_logits,
                    thresholds,
                    target,
                    loss_scale=1.0 if is_real else 0.0,
                )
                loss = losses["loss"] / accumulation
            loss.backward()
            flush = (slot + 1) % accumulation == 0 or slot + 1 == len(local_indices)
            if flush:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                reduced = torch.stack(
                    (
                        losses["loss"].detach(),
                        losses["report_loss"].detach(),
                        torch.tensor(float(is_real), device=device),
                    )
                )
                if world > 1:
                    dist.all_reduce(reduced)
                    reduced /= world
                if rank == 0 and (update == 1 or update % 10 == 0):
                    print(
                        f"epoch={epoch + 1} update={update} "
                        f"loss={reduced[0]:.4f} report={reduced[1]:.4f} "
                        f"real_fraction={reduced[2]:.3f}",
                        flush=True,
                    )
                checkpoint_every = int(config["training"]["checkpoint_every"])
                if checkpoint_every and update % checkpoint_every == 0:
                    save_checkpoint(
                        output_dir / f"checkpoint-{update:08d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        label_names,
                        epoch=epoch,
                        next_slot=slot + 1,
                        update=update,
                        rank=rank,
                        world=world,
                    )
                stop_requested = bool(_CHECKPOINT_AND_STOP) or bool(
                    args.max_updates and update >= args.max_updates
                )
                stop_tensor = torch.tensor(
                    int(stop_requested), dtype=torch.int32, device=device
                )
                if world > 1:
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                if stop_tensor.item():
                    stop = True
                    break
            if online_source is not None:
                verify_frozen_encoder(online_source)
            if any(parameter.grad is not None for parameter in mil_head.parameters()):
                raise RuntimeError("Frozen MIL head accumulated gradients")
        start_slot = 0
        if stop:
            break

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        config,
        label_names,
        epoch=epoch if "epoch" in locals() else 0,
        next_slot=(slot + 1) if "slot" in locals() else 0,
        update=update,
        rank=rank,
        world=world,
    )
    if rank == 0:
        print(f"Saved {output_dir / 'last.pt'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
