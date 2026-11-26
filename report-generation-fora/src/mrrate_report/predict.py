"""Frozen-MIL study logits for every study of the requested splits (one process per GPU, resumable).

Writes OUT/shard_r{rank:03d}_{k:05d}.npz (uids, splits, logits [S, 74] float32, n_tokens) and OUT/classes.json.
Already written studies are skipped on restart. Validation studies are checked against the MIL run's own validation
probabilities when the bundle holds them.

torchrun --nproc-per-node=8 -m mrrate_report.predict --bundle B --data-folder D --splits-csv S --out O
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .mil import MRVisualStream
from .schema import label_order_sha256


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="MIL bundle (backbone_release/, checkpoint/, code/, dfw_mil/)")
    ap.add_argument("--data-folder", required=True, help="MR-RATE mri_atlas root (batchXX/<uid>/atlas_img/*.nii.gz)")
    ap.add_argument("--splits-csv", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flush-every", type=int, default=128)
    ap.add_argument("--max-studies", type=int, default=0, help="smoke tests only")
    a = ap.parse_args(argv)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = MRVisualStream(a.bundle, device)
    if rank == 0:
        (out / "classes.json").write_text(json.dumps({"classes": stream.classes,
                                                     "label_order_sha256": label_order_sha256(stream.classes),
                                                     "mil_checkpoint": stream.checkpoint}))
    reference = {}
    summary = Path(a.bundle) / "checkpoint/epoch_004.json"
    if summary.is_file():
        reference = {r["study_uid"]: np.asarray(r["probabilities"])
                     for r in json.loads(summary.read_text())["predictions"]}
    from mr_dino.data import discover_raw_aligned
    want = set(a.splits.split(","))
    with open(a.splits_csv, newline="") as handle:
        split_of = {r["study_uid"]: r["split"] for r in csv.DictReader(handle) if r["split"] in want}
    studies = sorted(discover_raw_aligned(a.data_folder, set(split_of), stream.meta["args"]["space"]),
                     key=lambda s: s["study_uid"])
    if a.max_studies:
        studies = studies[: a.max_studies]
    done = set()
    for f in glob.glob(str(out / "shard_r*.npz")):
        done.update(map(str, np.load(f)["uids"]))
    mine = [s for i, s in enumerate(studies) if i % world == rank and s["study_uid"] not in done]
    k = len(glob.glob(str(out / f"shard_r{rank:03d}_*.npz")))
    print(json.dumps(dict(rank=rank, world=world, studies=len(studies), already_done=len(done), mine=len(mine))),
          flush=True)
    buffer = {"uids": [], "splits": [], "logits": [], "n_tokens": []}

    def flush():
        nonlocal k
        if not buffer["uids"]:
            return
        tmp = out / f".shard_r{rank:03d}_{k:05d}.tmp.npz"
        np.savez(tmp, uids=np.array(buffer["uids"]), splits=np.array(buffer["splits"]),
                 logits=np.stack(buffer["logits"]).astype(np.float32), n_tokens=np.array(buffer["n_tokens"], np.int64))
        os.replace(tmp, out / f"shard_r{rank:03d}_{k:05d}.npz")
        k += 1
        for v in buffer.values():
            v.clear()

    worst, t0 = 0.0, time.time()
    for n, study in enumerate(mine):
        uid = study["study_uid"]
        try:
            enc = stream.encode(study, stream.prepare(study))
        except Exception as error:   # one unreadable study must not stop the pass; it is logged and skipped
            print(json.dumps(dict(uid=uid, error=repr(error)[:300])), flush=True)
            continue
        logits = enc.logits.cpu().numpy()
        if uid in reference:
            worst = max(worst, float(np.abs(1 / (1 + np.exp(-logits)) - reference[uid]).max()))
        buffer["uids"].append(uid)
        buffer["splits"].append(split_of[uid])
        buffer["logits"].append(logits)
        buffer["n_tokens"].append(enc.total_tokens)
        del enc
        if len(buffer["uids"]) >= a.flush_every:
            flush()
        if n % 50 == 0:
            print(json.dumps(dict(rank=rank, done=n + 1, of=len(mine), s_per_study=round((time.time() - t0) / (n + 1), 2),
                                  val_parity_max_abs_dprob=round(worst, 5))), flush=True)
    flush()
    print(json.dumps(dict(rank=rank, finished=len(mine), val_parity_max_abs_dprob=round(worst, 5))), flush=True)


if __name__ == "__main__":
    main()
