"""Offline preprocessing: bake the dataloader's per-volume pipeline into .npz.

The coregistered/atlas NIfTIs are large (some ~2 GB) and slow to read + resample
on the fly, which starves the GPUs. This script runs the *exact* same per-volume
transform the dataloader uses — RAS reorient -> resample -> normalize -> crop/pad
(see data.preprocess_nii) — once, ahead of time, and writes one compact .npz per
subject. Training then reads those directly with `--use_preprocessed`.

It is deliberately independent of any dataloader/training decision:
  - Subject discovery uses the shared data.discover_subjects() (report/split
    agnostic), so it preprocesses every subject present in --data_folder.
  - The output is keyed only by the preprocessing config (space, spacing, shape,
    posterior shift, normalizer), recorded in a manifest the loader checks.

Output layout (consumed by MRReportDataset / MRReportDatasetInfer):

    <out_dir>/<space>/_manifest.json          # the config below
    <out_dir>/<space>/<study_uid>.npz         # volumes: [N, D, H, W] (float16)

Each .npz holds one array `volumes` of shape [N, D, H, W] = all of a subject's
volumes already preprocessed and stacked, in the same order as the sorted NIfTI
filenames. The loader casts to bfloat16 and adds the channel dim -> [N,1,D,H,W],
matching the live path exactly.

Resume-safe: existing .npz are skipped unless --overwrite. Shardable across
array jobs / nodes with --num_shards / --shard_index.

Example:
    python scripts/preprocess_volumes.py \
        --data_folder /path/to/MR-RATE-coreg/mri \
        --out_dir     /path/to/preprocessed \
        --space       coreg_space \
        --normalizer  zscore \
        --num_workers 8
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data import (
    CACHE_MANIFEST_NAME,
    NORMALIZERS,
    build_cache_manifest,
    discover_subjects,
    preprocess_nii,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute MR-RATE preprocessed volumes as per-subject .npz."
    )
    p.add_argument("--data_folder", required=True,
                   help="Raw MR data folder (same as training --data_folder).")
    p.add_argument("--out_dir", required=True,
                   help="Destination root. Files go to <out_dir>/<space>/.")
    p.add_argument("--space", default="native_space",
                   choices=["native_space", "coreg_space", "atlas_space"],
                   help="Which image space to preprocess (default: native_space).")

    # Preprocessing config — MUST match training to be reusable. Defaults mirror
    # MRReportDataset's defaults.
    p.add_argument("--normalizer", default="zscore",
                   choices=list(NORMALIZERS.keys()))
    p.add_argument("--normalizer_kwargs", type=str, default="{}",
                   help='JSON dict of kwargs for the normalizer, e.g. for '
                        'percentile: \'{"lower_percentile": 1.0, '
                        '"upper_percentile": 99.0}\'. Must match training. '
                        'Default: {} (normalizer defaults).')
    p.add_argument("--target_spacing", type=float, nargs=3,
                   default=(1.0, 0.5, 0.5), metavar=("D", "H", "W"),
                   help="Target voxel spacing in mm (D H W). Default: 1.0 0.5 0.5")
    p.add_argument("--target_shape", type=int, nargs=3,
                   default=(256, 384, 384), metavar=("D", "H", "W"),
                   help="Target volume shape (D H W). Default: 256 384 384")
    p.add_argument("--posterior_shift_mm", type=float, default=15.0,
                   help="Posterior shift on W (AP) axis in mm (default: 15.0).")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                   help="Stored dtype (default: float16, ~2x smaller on disk).")
    p.add_argument("--compress", action="store_true",
                   help="Use np.savez_compressed (smaller files, slower reads). "
                        "Default: uncompressed, optimized for fast training reads.")

    # Throughput / orchestration
    p.add_argument("--num_workers", type=int, default=4,
                   help="Parallel subject workers (process pool). Default: 4.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess subjects whose .npz already exists.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total shards for splitting across jobs/nodes.")
    p.add_argument("--shard_index", type=int, default=0,
                   help="This job's shard index in [0, num_shards).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many subjects (debugging).")
    return p.parse_args()


def _process_subject(args_tuple):
    """Worker: preprocess one subject's volumes and write its .npz.

    Returns (subject_id, status, n_volumes, message). status in
    {"written", "skipped", "empty", "error"}.
    """
    (subject_id, image_paths, out_path, target_spacing, target_shape,
     posterior_shift_voxels, normalizer_name, normalizer_kwargs, dtype,
     compress, overwrite) = args_tuple

    if os.path.exists(out_path) and not overwrite:
        return (subject_id, "skipped", 0, "")

    # Build a fresh normalizer per worker (cheap; avoids cross-process sharing).
    normalizer_obj = NORMALIZERS[normalizer_name](**(normalizer_kwargs or {}))

    try:
        vols = []
        for path in image_paths:
            arr = preprocess_nii(
                path, target_spacing, target_shape,
                posterior_shift_voxels, normalizer_obj,
            )  # float32 [D, H, W]
            vols.append(arr.astype(dtype, copy=False))
        if not vols:
            return (subject_id, "empty", 0, "no volumes")

        stacked = np.stack(vols, axis=0)  # [N, D, H, W]

        # Atomic write: tmp then rename, so an interrupted job never leaves a
        # half-written .npz that a later run would treat as complete.
        tmp_path = out_path + ".tmp.npz"
        save = np.savez_compressed if compress else np.savez
        save(tmp_path, volumes=stacked)
        # np.savez appends .npz if missing; tmp already ends in .npz so no double ext.
        os.replace(tmp_path, out_path)
        return (subject_id, "written", stacked.shape[0], "")
    except Exception as e:  # noqa: BLE001 — report and continue with other subjects
        # Clean any partial tmp
        tmp_path = out_path + ".tmp.npz"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return (subject_id, "error", 0, f"{type(e).__name__}: {e}")


def main():
    args = parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        print(f"ERROR: shard_index must be in [0, {args.num_shards}); "
              f"got {args.shard_index}", file=sys.stderr)
        sys.exit(2)

    target_spacing = tuple(args.target_spacing)
    target_shape = tuple(args.target_shape)
    posterior_shift_voxels = int(round(args.posterior_shift_mm / target_spacing[2]))
    try:
        normalizer_kwargs = json.loads(args.normalizer_kwargs)
    except json.JSONDecodeError as e:
        print(f"ERROR: --normalizer_kwargs is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(normalizer_kwargs, dict):
        print("ERROR: --normalizer_kwargs must be a JSON object (dict).", file=sys.stderr)
        sys.exit(2)
    # Fail fast if the kwargs don't fit the chosen normalizer (don't discover it
    # mid-run after reading gigabytes).
    try:
        NORMALIZERS[args.normalizer](**normalizer_kwargs)
    except TypeError as e:
        print(f"ERROR: --normalizer_kwargs {normalizer_kwargs} invalid for "
              f"normalizer '{args.normalizer}': {e}", file=sys.stderr)
        sys.exit(2)

    space_dir = os.path.join(args.out_dir, args.space)
    os.makedirs(space_dir, exist_ok=True)

    # --- Discover subjects (report/split agnostic) ---
    print(f"[preprocess] Discovering subjects under {args.data_folder} "
          f"(space={args.space}) ...", flush=True)
    subjects = discover_subjects(args.data_folder, args.space)
    print(f"[preprocess] Found {len(subjects)} subjects with volumes.", flush=True)
    if not subjects:
        print("[preprocess] Nothing to do.", flush=True)
        return

    # Deterministic shard split (subjects already sorted by discover_subjects)
    if args.num_shards > 1:
        subjects = subjects[args.shard_index::args.num_shards]
        print(f"[preprocess] Shard {args.shard_index}/{args.num_shards}: "
              f"{len(subjects)} subjects.", flush=True)
    if args.limit is not None:
        subjects = subjects[:args.limit]
        print(f"[preprocess] Limited to {len(subjects)} subjects.", flush=True)

    # --- Write / verify the manifest (shard 0 writes; others reconcile) ---
    manifest = build_cache_manifest(
        args.space, target_spacing, target_shape, args.posterior_shift_mm,
        args.normalizer, normalizer_kwargs, args.dtype,
    )
    manifest_path = os.path.join(space_dir, CACHE_MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        # Tolerate concurrent shards racing to create it; last writer wins with
        # identical content, so this is safe.
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[preprocess] Wrote manifest -> {manifest_path}", flush=True)
    else:
        with open(manifest_path) as f:
            existing = json.load(f)
        mismatched = {k: (existing.get(k), manifest[k])
                      for k in manifest if k != 'dtype' and existing.get(k) != manifest[k]}
        if mismatched:
            print(f"ERROR: existing manifest at {manifest_path} disagrees with "
                  f"requested config: {mismatched}. Use a different --out_dir.",
                  file=sys.stderr)
            sys.exit(3)

    # --- Build work list ---
    tasks = []
    for sub in subjects:
        out_path = os.path.join(space_dir, f"{sub['subject_id']}.npz")
        tasks.append((
            sub['subject_id'], sub['image_paths'], out_path,
            target_spacing, target_shape, posterior_shift_voxels,
            args.normalizer, normalizer_kwargs, args.dtype,
            args.compress, args.overwrite,
        ))

    # --- Run ---
    counts = {"written": 0, "skipped": 0, "empty": 0, "error": 0}
    n_vol_total = 0
    t0 = time.time()

    def _handle(result):
        sid, status, n, msg = result
        counts[status] = counts.get(status, 0) + 1
        if status == "written":
            return n
        if status in ("error", "empty"):
            print(f"  [{status}] {sid}: {msg}", flush=True)
        return 0

    done = 0
    total = len(tasks)
    if args.num_workers <= 1:
        for t in tasks:
            n_vol_total += _handle(_process_subject(t))
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[preprocess] {done}/{total} subjects "
                      f"({counts['written']} written, {counts['skipped']} skipped, "
                      f"{counts['error']} errors), {time.time()-t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_process_subject, t) for t in tasks]
            for fut in as_completed(futures):
                n_vol_total += _handle(fut.result())
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"[preprocess] {done}/{total} subjects "
                          f"({counts['written']} written, {counts['skipped']} skipped, "
                          f"{counts['error']} errors), {time.time()-t0:.0f}s", flush=True)

    print(f"\n[preprocess] DONE in {time.time()-t0:.0f}s -> {space_dir}")
    print(f"  written={counts['written']}  skipped={counts['skipped']}  "
          f"empty={counts['empty']}  errors={counts['error']}  "
          f"volumes_written={n_vol_total}")
    if counts["error"]:
        print(f"  WARNING: {counts['error']} subjects failed; rerun (without "
              f"--overwrite) to retry just the missing ones.")


if __name__ == "__main__":
    main()
