"""Re-label an existing extract_features.py output for a NEW label set without
re-encoding.

The frozen encoder never sees labels, so `features_<split>.npy` and
`subject_ids_<split>.txt` are identical across label schemes — only
`labels_<split>.npy` and `label_names.json` depend on the label CSV. Running
`extract_features.py` again just to change labels would repeat the expensive 3D
encoder pass for nothing. This script reuses the cached features and rebuilds
only the label artifacts.

It writes a new feature dir that:
  - symlinks (or copies, with --copy) `features_<split>.npy` and
    `subject_ids_<split>.txt` from the source extraction, and
  - writes fresh `labels_<split>.npy` (in the SAME subject order as the source)
    and `label_names.json` from --labels_file.

Then train as usual:  python scripts/linear_probe.py --features_dir <out_dir>

Example — extract features once, then probe another labels CSV with no re-encode:

    # one-time encode
    for S in train val test; do
      python scripts/extract_features.py ... \
          --labels_file scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv \
          --splits_csv  scripts/eval_labels/splits_merged_majority/splits.csv \
          --split $S --out_dir ./lp_features
    done

    # probe any other labels CSV on the SAME cached features
    python scripts/relabel_features.py \
        --features_dir ./lp_features \
        --labels_file  /path/to/other_labels.csv \
        --out_dir      ./lp_features_other
    python scripts/linear_probe.py --features_dir ./lp_features_other --results_dir ./lp_results_other
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np


def load_labels(csv_path: Path):
    """Return (dict study_uid -> float32 vector, list of class column names)."""
    with open(csv_path) as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        id_col = "study_uid" if "study_uid" in fields else "subject_id"
        cols = [c for c in fields if c != id_col]
        d = {}
        for row in r:
            d[row[id_col]] = np.array([float(row[c]) for c in cols], dtype=np.float32)
    return d, cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True,
                    help="Existing extract_features.py output (features_*.npy + subject_ids_*.txt).")
    ap.add_argument("--labels_file", required=True,
                    help="New labels CSV: study_uid + per-class binary columns.")
    ap.add_argument("--out_dir", required=True, help="New feature dir to create.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--copy", action="store_true",
                    help="Copy features instead of symlinking (default: symlink).")
    a = ap.parse_args()

    src = Path(a.features_dir)
    out = Path(a.out_dir)
    if out.resolve() == src.resolve():
        raise SystemExit("--out_dir must differ from --features_dir (would clobber features).")
    out.mkdir(parents=True, exist_ok=True)

    labels, cols = load_labels(Path(a.labels_file))
    (out / "label_names.json").write_text(
        json.dumps(cols, indent=2, ensure_ascii=False) + "\n")
    print(f"label_names.json: {len(cols)} classes")

    n_done = 0
    for split in a.splits:
        sid_path = src / f"subject_ids_{split}.txt"
        feat_path = src / f"features_{split}.npy"
        if not (sid_path.exists() and feat_path.exists()):
            print(f"[skip] {split}: no features/subject_ids in {src}")
            continue
        sids = sid_path.read_text().strip().splitlines()
        missing = [s for s in sids if s not in labels]
        if missing:
            raise SystemExit(
                f"{split}: {len(missing)} subject(s) absent from {a.labels_file}; "
                f"cannot relabel (would misalign features). e.g. {missing[:5]}")
        Y = np.stack([labels[s] for s in sids], axis=0).astype(np.float32)
        np.save(out / f"labels_{split}.npy", Y)
        for name in (f"features_{split}.npy", f"subject_ids_{split}.txt"):
            dst = out / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if a.copy:
                shutil.copy(src / name, dst)
            else:
                os.symlink((src / name).resolve(), dst)
        link = "copied" if a.copy else "linked"
        print(f"[ok] {split}: {Y.shape[0]} subjects x {Y.shape[1]} classes  (features {link})")
        n_done += 1

    if not n_done:
        raise SystemExit(f"No splits found in {src}. Run extract_features.py there first.")
    print(f"\nDone -> {out}\n  python scripts/linear_probe.py --features_dir {out}")


if __name__ == "__main__":
    main()
