# =======================================================================
# Merge per-rank classification JSONs into a single CSV
#
# Reads labels_rank_*.json files from the output directory and produces
# a CSV with columns: study_uid, PathologyA, PathologyB, ... (0/1 values)
# =======================================================================

import argparse
import csv
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-rank classification JSONs into a single CSV"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing labels_rank_*.json files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "labels_rank_*.json")))
    if not files:
        print(f"No labels_rank_*.json files found in {args.input_dir}")
        return

    # Collect all results
    all_rows = []
    pathology_names = None
    total_stats = {
        "json_ok": 0, "cot_fallback": 0, "retries": 0,
        "disagreements": 0, "verified": 0, "flipped_to_absent": 0,
    }

    for f_path in files:
        with open(f_path) as f:
            data = json.load(f)

        if pathology_names is None:
            pathology_names = list(data["results"][0]["labels"].keys())

        for k, v in data["metadata"]["stats"].items():
            if k in total_stats:
                total_stats[k] += v

        for r in data["results"]:
            row = [r["study_uid"]] + [r["labels"].get(p, 0) for p in pathology_names]
            all_rows.append(row)

    # Sort by study_uid for deterministic output
    all_rows.sort(key=lambda x: x[0])

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_uid"] + pathology_names)
        writer.writerows(all_rows)

    # Print summary
    from collections import Counter
    present_counts = Counter()
    for row in all_rows:
        for i, p in enumerate(pathology_names):
            if row[i + 1] == 1:
                present_counts[p] += 1

    print(f"Merged {len(files)} rank files -> {len(all_rows)} reports")
    print(f"Output: {args.output}")
    print(f"\nAggregate stats: {total_stats}")
    print(f"\nPathology prevalence:")
    for p, c in present_counts.most_common():
        print(f"  {c:6d} ({100 * c / len(all_rows):5.1f}%) | {p}")


if __name__ == "__main__":
    main()
