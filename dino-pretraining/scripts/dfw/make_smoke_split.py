"""Write a small patient-disjoint subset of the MR-RATE splits for the MIL smoke test.

Keeps only labelled studies that have an image folder for the chosen space, so the
subset is a complete cohort and MIL runs without --allow-incomplete-data.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-csv", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--data-folder", required=True)
    p.add_argument("--space", default="atlas_space")
    p.add_argument("--train", type=int, default=96)
    p.add_argument("--val", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.labels_csv, newline="") as handle:
        labelled = {row["study_uid"] for row in csv.DictReader(handle)}
    with open(args.splits_csv, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["study_uid"] in labelled]
    img_dir = args.space.removesuffix("_space") + "_img"
    rng = random.Random(args.seed)
    chosen = []
    for split, n in (("train", args.train), ("val", args.val)):
        pool = [r for r in rows if r["split"] == split]
        rng.shuffle(pool)
        picked = []
        for r in pool:
            folder = Path(args.data_folder) / r["batch_id"] / r["study_uid"] / img_dir
            if folder.is_dir() and any(folder.glob("*.nii.gz")):
                picked.append(r)
            if len(picked) == n:
                break
        if len(picked) < n:
            raise SystemExit(f"only {len(picked)} usable {split} studies")
        chosen += picked
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(chosen)
    print(f"wrote {out}: {args.train} train + {args.val} val studies")


if __name__ == "__main__":
    main()
