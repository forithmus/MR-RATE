"""
Generic shard merger for parallel SLURM jobs.
Merges per-rank CSV outputs into a single file with deduplication and stats.

Usage:
    python merge_shards.py --shard_dir <dir> --output <file> [--shard_prefix <prefix>] [--dedup_col <col>]
"""

import argparse
import glob
import os
import pandas as pd


def merge_shards(shard_dir, output_file, shard_prefix="*_rank_", dedup_col="AccessionNo"):
    files = sorted(glob.glob(os.path.join(shard_dir, f"{shard_prefix}*.csv")))
    print(f"Found {len(files)} shard files in {shard_dir}")

    dfs = []
    for f in files:
        d = pd.read_csv(f, encoding="utf-8-sig")
        print(f"  {os.path.basename(f)}: {len(d)} rows")
        dfs.append(d)

    merged = pd.concat(dfs, ignore_index=True)

    if dedup_col and dedup_col in merged.columns:
        before = len(merged)
        merged.drop_duplicates(subset=dedup_col, keep="last", inplace=True)
        if before != len(merged):
            print(f"  Deduplicated: {before} -> {len(merged)}")

    # Print status distribution if parse_status or verdict column exists
    for status_col in ["parse_status", "verdict", "qc_status"]:
        if status_col in merged.columns:
            counts = merged[status_col].value_counts()
            print(f"\n{status_col} distribution:")
            for k, v in counts.items():
                print(f"  {k}: {v} ({100*v/len(merged):.1f}%)")

    merged.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(merged)} rows to {output_file}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel SLURM job shards")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing shard CSVs")
    parser.add_argument("--output", type=str, required=True, help="Output merged CSV path")
    parser.add_argument("--shard_prefix", type=str, default="*_rank_", help="Shard filename prefix pattern")
    parser.add_argument("--dedup_col", type=str, default="AccessionNo", help="Column for deduplication")
    args = parser.parse_args()

    merge_shards(args.shard_dir, args.output, args.shard_prefix, args.dedup_col)
