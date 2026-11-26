"""Resumable train-only feature normalization, same statistics as ``mr_dino.vlm normalize`` (online).

With ``--train-fraction`` the statistics use the same nested patient subset as
``mil_fast --train-fraction`` and record it, so a subset run never sees the other patients.

Each rank accumulates float64 sums over ``train_ids[rank::world]`` in sorted study order and
saves its partial sums to ``<output>.partial/rank<R>.pt`` every ``--save-every`` studies and
before the Slurm deadline, then writes ``<output dir>/REQUEUE_REQUESTED`` and exits 75.
The final file holds the mean, RMS, cohort size and token count.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from mr_dino.data import discover_raw_aligned
from mr_dino.mil_stream import LastBlockAdapter
from mr_dino.release import load_release

from .fast import Deadline, Prefetcher, cached_study_factory, emit, load_study
from .subset import subset_train_ids

REQUEUE = 75


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True)
    p.add_argument("--data-folder", required=True)
    p.add_argument("--splits-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-incomplete-data", action="store_true")
    p.add_argument("--tile-batch", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--deadline-every", type=int, default=25,
                   help="studies between collective deadline checks; each check waits for the slowest rank")
    args = p.parse_args()

    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if world > 1 else 0
    output = Path(args.output)
    if output.exists():
        if rank == 0:
            print(f"{output} already exists; nothing to do", flush=True)
        return 0

    backbone, metadata = load_release(args.release, device)
    adapter = LastBlockAdapter(backbone).to(device)
    with open(args.splits_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    membership = {r["study_uid"]: r["split"] for r in rows}
    patients = {s: {r["patient_uid"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Patient split leakage")
    studies = {s["study_uid"]: s for s in discover_raw_aligned(args.data_folder, set(membership), metadata["args"]["space"])}
    ids = sorted(u for u in studies if membership.get(u) == "train")
    patient_of = {r["study_uid"]: r["patient_uid"] for r in rows}
    ids = subset_train_ids(ids, patient_of, args.train_fraction, args.subset_seed)
    missing = [u for u in membership if membership[u] == "train" and u not in studies]
    if missing and not args.allow_incomplete_data:
        raise ValueError(f"Incomplete training normalization cohort: {len(missing)} missing MRI studies")
    mine = ids[rank::world]

    partial_dir = output.with_name(output.name + ".partial")
    partial = partial_dir / f"rank{rank:04d}.pt"
    sums = torch.zeros(adapter.dim, device=device, dtype=torch.float64)
    squared = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros_like(squared)
    done = 0
    if partial.exists():
        state = torch.load(partial, map_location=device, weights_only=False)
        if state["world"] != world or state["release_config"] != metadata["config"] or state["ids"] != len(ids):
            raise ValueError("Partial normalization belongs to another run")
        sums, squared, count, done = state["sums"], state["squared"], state["count"], state["done"]
    partial_dir.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = partial.with_suffix(".pending")
        torch.save({"world": world, "release_config": metadata["config"], "ids": len(ids), "done": done,
                    "sums": sums, "squared": squared, "count": count}, tmp)
        tmp.replace(partial)

    if rank == 0:
        print({"event": "normalize_start", "train_studies": len(ids), "world": world, "resumed_at": done}, flush=True)
    deadline = Deadline(device)
    steps = -(-len(ids) // world)
    prefetch = Prefetcher([studies[u] for u in mine], lambda s: load_study(s, metadata), args.prefetch)
    with torch.no_grad():
        for k in range(done, steps):
            if k < len(mine):
                volumes = prefetch.get(k)
                start = time.perf_counter()
                factory = cached_study_factory(studies[mine[k]], volumes, backbone, adapter, metadata, device, args.tile_batch)
                for y, _, _, _ in factory():
                    sums += y.double().sum(0)
                    squared += y.double().square().sum()
                    count += len(y)
                emit(event="norm_study", seconds=time.perf_counter() - start)
                done = k + 1
            else:
                done = k + 1
            last = k + 1 == steps
            if done % args.save_every == 0 or last:
                save()
            if not last and done % args.deadline_every == 0 and deadline.reached():
                save()
                prefetch.close()
                if rank == 0:
                    (output.parent / "REQUEUE_REQUESTED").touch()
                    print({"event": "normalize_deadline", "done_per_rank": done, "of": steps}, flush=True)
                return REQUEUE
    prefetch.close()
    if world > 1:
        for value in (sums, squared, count):
            dist.all_reduce(value)
    if count <= 0:
        raise ValueError("Empty training normalization population")
    mean = sums / count
    rms = ((squared / count - mean.square().sum()) / adapter.dim).clamp_min(1e-12).sqrt()
    if rank == 0:
        state = {"mean": mean.float().cpu(), "rms": float(rms), "release_config": metadata["config"],
                 "split": "train", "studies": len(ids), "tokens": int(count),
                 "train_fraction": args.train_fraction, "subset_seed": args.subset_seed}
        pending = output.with_suffix(".pending")
        torch.save(state, pending)
        pending.replace(output)
        print({"event": "normalization_complete", "studies": len(ids), "tokens": int(count), "rms": float(rms)}, flush=True)
    emit(event="normalize_done", studies=len(mine))
    return 0


if __name__ == "__main__":
    code = main()
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(code)
