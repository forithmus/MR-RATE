"""Online multi-node ``mr_dino.mil train``, resumable mid-epoch, with the MIL ablations.

Model, objective, per-epoch study order, gradient averaging, optimizer, validation and the
``epoch_NNN.{pt,json}`` outputs are those of ``mr_dino.mil train``.
  * Study features come from ``fast`` (prefix computed once per study, next studies loaded in
    the background).
  * ``latest.pt`` holds the full training state plus the position inside the epoch; it is
    written every ``--save-every-steps`` optimizer steps, before the Slurm deadline, and before
    validation when less than ``--val-reserve`` seconds remain. A run restarted with the same
    ``--out`` continues at the next optimizer step. A checkpoint-and-exit writes
    ``<out>/REQUEUE_REQUESTED`` (torchrun reports the 75 exit code as 1).

Ablations
  * ``--train-fraction f`` trains on a nested patient-level subset of the training split
    (``subset``); the normalization must come from the same subset. Validation is unchanged.
  * ``--fusion cross_sequence`` adds attention among the sequences at each atlas token
    (``fusion.CrossSequenceAttention``) between the last block and the MIL head.
  * A modality-channel release (``mr_dino.export_release`` of a ``--input-channels modality``
    DINO run) is used as is: its inputs are modality stacks instead of single sequences.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from mr_dino.data import discover_raw_aligned
from mr_dino.geometry import position_features
from mr_dino.labels import DEFAULT_SCHEMA, read_final_labels
from mr_dino.mil_head import SparseClassifyThenAggregate
from mr_dino.mil_stream import LastBlockAdapter, backward, predict
from mr_dino.probe import auroc
from mr_dino.release import load_release

from .fast import Deadline, Prefetcher, cached_study_factory, cross_sequence_factory, emit, load_study
from .fusion import CrossSequenceAttention
from .subset import subset_train_ids

REQUEUE = 75


def parse_args():
    t = argparse.ArgumentParser()
    t.add_argument("--release", required=True)
    t.add_argument("--data-folder", required=True)
    t.add_argument("--normalization", required=True)
    t.add_argument("--position-freqs", type=int, default=4)
    t.add_argument("--position-scale", type=float, default=1.)
    t.add_argument("--allow-incomplete-data", action="store_true")
    t.add_argument("--labels-csv", required=True)
    t.add_argument("--splits-csv", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--label-schema", default=str(DEFAULT_SCHEMA))
    t.add_argument("--head-lr", type=float, required=True)
    t.add_argument("--block-lr", type=float, required=True)
    t.add_argument("--weight-decay", type=float, default=.01)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--tile-batch", type=int, default=4)
    t.add_argument("--prefetch", type=int, default=2)
    t.add_argument("--save-every-steps", type=int, default=200)
    t.add_argument("--val-reserve", type=float, default=3600.)
    t.add_argument("--stop-after-steps", type=int, default=0, help="test only: checkpoint and exit 75 after N steps")
    t.add_argument("--train-fraction", type=float, default=1.0,
                   help="fraction of training patients (nested subsets for one --subset-seed)")
    t.add_argument("--subset-seed", type=int, default=0)
    t.add_argument("--fusion", choices=("none", "cross_sequence"), default="none")
    t.add_argument("--fusion-depth", type=int, default=1)
    t.add_argument("--fusion-heads", type=int, default=8)
    t.add_argument("--fusion-mlp-ratio", type=float, default=2.0)
    t.add_argument("--fusion-lr", type=float, default=1e-4)
    t.add_argument("--fusion-recompute", action=argparse.BooleanOptionalAction, default=True,
                   help="recompute the last block per sequence in the backward pass to save memory")
    return t.parse_args()


def train(args) -> int:
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if world > 1 else 0
    torch.manual_seed(args.seed)
    model, meta = load_release(args.release, device)
    adapter = LastBlockAdapter(model).to(device)
    names, labels = read_final_labels(args.labels_csv, args.label_schema)
    normalizer = torch.load(args.normalization, map_location=device, weights_only=False)
    if normalizer["release_config"] != meta["config"] or normalizer["split"] != "train":
        raise ValueError("Wrong normalization provenance")
    if (float(normalizer.get("train_fraction", 1.0)) != args.train_fraction
            or (args.train_fraction < 1 and normalizer.get("subset_seed") != args.subset_seed)):
        raise ValueError("Normalization was computed on another training subset")
    mean = normalizer["mean"].float()
    rms = float(normalizer["rms"])
    if mean.shape != (adapter.dim,) or not math.isfinite(rms) or rms <= 0:
        raise ValueError("Invalid MRI normalization")
    head = SparseClassifyThenAggregate(adapter.dim + 6 * args.position_freqs, len(names), use_norm=False,
                                       checkpoint_chunks=False).to(device)
    head.route_bias_raw = torch.nn.Parameter(torch.full((len(names),), 2., device=device))
    fusion = None
    if args.fusion == "cross_sequence":
        fusion = CrossSequenceAttention(adapter.dim, args.fusion_depth, args.fusion_heads, args.fusion_mlp_ratio).to(device)
    params = list(adapter.parameters()) + list(head.parameters()) + (list(fusion.parameters()) if fusion else [])
    groups = [{"params": adapter.parameters(), "lr": args.block_lr},
              {"params": [p for n, p in head.named_parameters() if n != "route_bias_raw"], "lr": args.head_lr},
              {"params": [head.route_bias_raw], "lr": args.head_lr, "weight_decay": 0.}]
    if fusion is not None:
        groups.append({"params": fusion.parameters(), "lr": args.fusion_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    with open(args.splits_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = {r["study_uid"]: r["split"] for r in rows}
    patient = {r["study_uid"]: r["patient_uid"] for r in rows}
    patient_sets = {s: {patient[u] for u in splits if splits[u] == s} for s in ("train", "val", "test")}
    if any(patient_sets[a] & patient_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Patient leakage between MR-RATE splits")
    studies = {}
    for study in discover_raw_aligned(args.data_folder, set(splits), meta["args"]["space"]):
        if study["study_uid"] in studies:
            raise ValueError("Duplicate online study")
        studies[study["study_uid"]] = study

    def make_factory(uid, volumes):
        if fusion is not None:
            fused = cross_sequence_factory(studies[uid], volumes, model, adapter, meta, device, args.tile_batch,
                                           recompute=args.fusion_recompute)

            def attended():
                for y, valid, kinds, coordinates in fused():
                    y = fusion((y - mean) / rms, valid, kinds)[valid]
                    coordinates = coordinates[:, None].expand(-1, valid.shape[1], -1)[valid]
                    yield torch.cat((y, position_features(coordinates, args.position_freqs) * args.position_scale), 1), None
            return attended
        raw = cached_study_factory(studies[uid], volumes, model, adapter, meta, device, args.tile_batch)

        def transformed():
            for y, mask, coordinates, _ in raw():
                yield torch.cat(((y - mean) / rms, position_features(coordinates, args.position_freqs) * args.position_scale), 1), mask
        return transformed

    train_ids = sorted(u for u in studies if splits.get(u) == "train" and u in labels)
    all_train = len(train_ids)
    train_ids = subset_train_ids(train_ids, patient, args.train_fraction, args.subset_seed)
    if rank == 0:
        print({"event": "cohort", "train_fraction": args.train_fraction, "subset_seed": args.subset_seed,
               "train_studies": len(train_ids), "of": all_train,
               "train_patients": len({patient[u] for u in train_ids})}, flush=True)
    val_ids = sorted(u for u in studies if splits.get(u) == "val" and u in labels)
    if not train_ids or not val_ids:
        raise ValueError("Both train and validation studies are required")
    missing = [u for u in labels if splits.get(u) in ("train", "val") and u not in studies]
    if missing and not args.allow_incomplete_data:
        raise ValueError(f"Incomplete MRI cohort: {len(missing)} labelled train/val studies missing; first={missing[:10]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "TRAINING_COMPLETE").exists():
        if rank == 0:
            print(f"{out} has TRAINING_COMPLETE; nothing to do", flush=True)
        return 0

    def state_dict(epoch_done):
        state = {"format": "mrdino_mil_v1", "epoch": epoch_done, "release_config": meta["config"], "classes": names,
                 "adapter": adapter.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                 "arguments": vars(args), "normalization": normalizer,
                 "architecture": {"dim": head.dim, "n_classes": len(names), "hidden_dim": 512, "mlp_hidden_dim": 384, "use_norm": False},
                 "train_studies": len(train_ids)}
        if fusion is not None:
            state["fusion"] = fusion.state_dict()
            state["architecture"]["fusion"] = {"kind": args.fusion, **fusion.config()}
        return state

    def save(path, state):
        pending = out / "checkpoint.pending"
        torch.save(state, pending)
        pending.replace(path)

    start_epoch, steps_done, phase, loss_sum, loss_count = 0, 0, "train", 0.0, 0
    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["release_config"] != meta["config"] or state["classes"] != names or state["world"] != world:
            raise ValueError("latest.pt belongs to another run (release, classes or world size differ)")
        if state["train_ids"] != len(train_ids) or state["seed"] != args.seed:
            raise ValueError("latest.pt was written for a different cohort or seed")
        if state["arguments"].get("fusion", "none") != args.fusion:
            raise ValueError("latest.pt was written with another --fusion")
        adapter.load_state_dict(state["adapter"])
        head.load_state_dict(state["head"])
        if fusion is not None:
            fusion.load_state_dict(state["fusion"])
        optimizer.load_state_dict(state["optimizer"])
        position = state["position"]
        start_epoch, steps_done, phase = position["epoch"], position["steps_done"], position["phase"]
        loss_sum, loss_count = position["loss_sum"], position["loss_count"]
        if rank == 0:
            print({"event": "resume", **position}, flush=True)

    def global_losses(losses_local):
        meter = torch.tensor([sum(losses_local), len(losses_local)], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(meter)
        return float(meter[0]), int(meter[1])

    def checkpoint_latest(epoch, steps, phase_name, total_sum, total_count):
        if rank == 0:
            state = state_dict(epoch)
            state.update(world=world, seed=args.seed, train_ids=len(train_ids),
                         position={"epoch": epoch, "steps_done": steps, "phase": phase_name,
                                   "loss_sum": total_sum, "loss_count": total_count})
            save(latest, state)
        if world > 1:
            dist.barrier(device_ids=[local])

    def objective(logits, target):
        known = torch.isfinite(target) & (target >= 0)
        if not bool(known.any()):
            raise ValueError("Study has no known labels")
        if not bool(((target[known] == 0) | (target[known] == 1)).all()):
            raise ValueError("Nonbinary pathology labels")
        return torch.nn.functional.binary_cross_entropy_with_logits(logits[known], target[known])

    deadline = Deadline(device)
    load = lambda uid: load_study(studies[uid], meta)
    session_steps = 0
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        steps = math.ceil(len(order) / world)
        order = order + [-1] * (steps * world - len(order))
        mine = order[rank::world]
        first = steps_done if epoch == start_epoch else 0
        if epoch == start_epoch and phase == "val":
            first = steps
        head.train()
        adapter.train()
        if fusion is not None:
            fusion.train()
        losses = []
        prefetch = Prefetcher([train_ids[i] if i >= 0 else None for i in mine], load, args.prefetch)
        for k in range(first, steps):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            index = mine[k]
            real = index >= 0
            if real:
                uid = train_ids[index]
                target = torch.tensor(labels[uid], device=device)
                volumes = prefetch.get(k)
                compute = time.perf_counter()
                loss, _ = backward(head, make_factory(uid, volumes), target, objective)
                losses.append(float(loss))
                emit(event="train_study", seconds=time.perf_counter() - compute,
                     max_mem_gib=torch.cuda.max_memory_allocated() / 2**30)
            if world > 1:
                count = torch.tensor(float(real), device=device)
                dist.all_reduce(count)
                for param in params:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    dist.all_reduce(param.grad)
                    param.grad.div_(count.clamp_min(1))
            torch.nn.utils.clip_grad_norm_(params, 3., error_if_nonfinite=True)
            optimizer.step()
            session_steps += 1
            if rank == 0:
                emit(event="step", seconds=time.perf_counter() - step_start, epoch=epoch + 1, step=k + 1, of=steps)
            done = k + 1
            stop_test = args.stop_after_steps and session_steps >= args.stop_after_steps
            if done < steps and (stop_test or deadline.reached()):
                part_sum, part_count = global_losses(losses)
                checkpoint_latest(epoch, done, "train", loss_sum + part_sum, loss_count + part_count)
                prefetch.close()
                if rank == 0:
                    (out / "REQUEUE_REQUESTED").touch()
                    print({"event": "checkpoint_and_requeue", "epoch": epoch + 1, "steps_done": done, "of": steps}, flush=True)
                return REQUEUE
            if done % args.save_every_steps == 0 and done < steps:
                part_sum, part_count = global_losses(losses)
                loss_sum, loss_count, losses = loss_sum + part_sum, loss_count + part_count, []
                checkpoint_latest(epoch, done, "train", loss_sum, loss_count)
        prefetch.close()
        part_sum, part_count = global_losses(losses)
        loss_sum, loss_count, losses = loss_sum + part_sum, loss_count + part_count, []
        if first < steps and deadline.reached(args.val_reserve):
            checkpoint_latest(epoch, steps, "val", loss_sum, loss_count)
            if rank == 0:
                (out / "REQUEUE_REQUESTED").touch()
                print({"event": "checkpoint_before_validation", "epoch": epoch + 1}, flush=True)
            return REQUEUE

        head.eval()
        adapter.eval()
        if fusion is not None:
            fusion.eval()
        predictions = []
        mine_val = val_ids[rank::world]
        prefetch = Prefetcher(mine_val, load, args.prefetch)
        for k, uid in enumerate(mine_val):
            volumes = prefetch.get(k)
            compute = time.perf_counter()
            probability = predict(head, make_factory(uid, volumes)).sigmoid()[0].cpu().tolist()
            emit(event="val_study", seconds=time.perf_counter() - compute)
            predictions.append({"study_uid": uid, "probabilities": probability})
        prefetch.close()
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, predictions)
            predictions = [r for shard in gathered for r in shard]
        if rank == 0:
            per_class = {}
            for c, name in enumerate(names):
                valid = [r for r in predictions if np.isfinite(labels[r["study_uid"]][c]) and labels[r["study_uid"]][c] >= 0]
                per_class[name] = auroc(np.array([labels[r["study_uid"]][c] for r in valid]),
                                        np.array([r["probabilities"][c] for r in valid])) if valid else None
            values = [v for v in per_class.values() if v is not None]
            result = {"epoch": epoch + 1, "train_loss": loss_sum / max(loss_count, 1),
                      "macro_auroc": float(np.mean(values)) if values else None,
                      "per_class_auroc": per_class, "predictions": predictions}
            (out / f"epoch_{epoch + 1:03d}.json").write_text(json.dumps(result, indent=2))
            state = state_dict(epoch + 1)
            state["route_masks_used"] = False
            save(out / f"epoch_{epoch + 1:03d}.pt", state)
            print({k: v for k, v in result.items() if k not in {"predictions", "per_class_auroc"}}, flush=True)
        loss_sum, loss_count, steps_done, phase = 0.0, 0, 0, "train"
        if epoch + 1 < args.epochs:
            checkpoint_latest(epoch + 1, 0, "train", 0.0, 0)
        elif world > 1:
            dist.barrier(device_ids=[local])
    if rank == 0:
        (out / "TRAINING_COMPLETE").touch()
    return 0


if __name__ == "__main__":
    code = train(parse_args())
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(code)
