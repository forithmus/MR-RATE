#!/usr/bin/env python3
"""
Merge per-rank labels_rank_*.json (from extract_neurovfm_dx_gemma.py) into a
single wide CSV: study_uid + one 0/1 column per NeuroVFM diagnosis.

Also writes an optional *_rationales.json keyed by study_uid if --rationales
is given.

Usage:
  python merge_labels.py --input_dir /path/to/output --output neurovfm_labels.csv
"""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Dir with labels_rank_*.json")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--rationales", default=None,
                    help="Optional: also dump per-study rationales JSON here")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "labels_rank_*.json")))
    if not files:
        raise SystemExit(f"No labels_rank_*.json in {args.input_dir}")

    diagnosis_keys = None
    rows = []            # list of (study_uid, labels dict)
    rationales_out = {}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        keys = d.get("metadata", {}).get("diagnosis_keys")
        if keys and diagnosis_keys is None:
            diagnosis_keys = keys
        for r in d.get("results", []):
            rows.append((r["study_uid"], r["labels"]))
            if args.rationales and r.get("rationales"):
                rationales_out[r["study_uid"]] = r["rationales"]

    if diagnosis_keys is None:
        # Fall back to union of label keys, sorted for stability.
        diagnosis_keys = sorted({k for _, lab in rows for k in lab})

    # De-dup study_uid (last write wins), preserve first-seen order.
    seen = {}
    order = []
    for uid, lab in rows:
        if uid not in seen:
            order.append(uid)
        seen[uid] = lab

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_uid"] + diagnosis_keys)
        for uid in order:
            lab = seen[uid]
            w.writerow([uid] + [int(lab.get(k, 0)) for k in diagnosis_keys])

    print(f"Wrote {len(order)} rows x {len(diagnosis_keys)} diagnoses "
          f"-> {args.output}")

    if args.rationales:
        with open(args.rationales, "w") as f:
            json.dump(rationales_out, f, ensure_ascii=False, indent=2)
        print(f"Wrote rationales for {len(rationales_out)} studies "
              f"-> {args.rationales}")


if __name__ == "__main__":
    main()
