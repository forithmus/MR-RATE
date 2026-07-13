"""
MR-RATE Fixed Registration Studies Downloader
=============================================
Re-downloads studies whose per-study zips were re-uploaded to the coreg and
atlas HuggingFace repos to correct data defects (truncated / empty / corrupt
archives). The corrected zips overwrite the previous ones, so any stale local
copy must be replaced.

The download is resume-friendly: each local zip is compared against the repo
version by LFS SHA-256, and only re-downloaded if it is missing or does not
match. Extracted study folders (`mri/<batch>/<uid>/`) are always removed, since
unzipping is a separate follow-up step and any partial extraction is unreliable.
So the intended workflow is:

    1. Run this script (safe to interrupt and re-run until it completes).
    2. Unzip the corrected zips (see docs/fixed_reg_studies.md).

The coreg and atlas repos are not symmetric, so the manifest lists each
derivative's affected studies separately.

JSON format (e.g. fixed_truncated_reg_study_ids.json)
-----------------------------------------------------
{
    "coreg": {"batch03": ["uid1", ...], ...},
    "atlas": {"batch03": ["uid1", ...], ...}
}

Usage
-----
    python download_fixed_reg_studies.py --json-path fixed_truncated_reg_study_ids.json --output-base /data/root --coreg --atlas
    python download_fixed_reg_studies.py --json-path fixed_corrupt_reg_study_ids.json --output-base /data/root --atlas --download-workers 16

Arguments
---------
    --json-path PATH      Path to the JSON manifest of fixed studies (nested by derivative).
    --output-base DIR     Root data directory (same as --output-base in download.py).
    --coreg               Re-download fixed coreg-space studies from Forithmus/MR-RATE-coreg. (default: disabled)
    --atlas               Re-download fixed atlas-space studies from Forithmus/MR-RATE-atlas. (default: disabled)
    --download-workers N  Concurrent download threads. (default: 8)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm


MRI_DERIVATIVES = {
    "coreg": ("Forithmus/MR-RATE-coreg", "_coreg", "MR-RATE-coreg"),
    "atlas": ("Forithmus/MR-RATE-atlas", "_atlas", "MR-RATE-atlas"),
}


def _sha256(path: Path) -> str:
    """Stream-hash a file to its SHA-256 hex digest (matches the git-LFS oid)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_meta(api: HfApi, repo_id: str, hf_paths: list) -> dict:
    """Map hf_path -> (sha256, size) for LFS files in the repo. Empty on failure (falls back to unconditional re-download)."""
    meta = {}
    try:
        for i in range(0, len(hf_paths), 50):
            for info in api.get_paths_info(repo_id, hf_paths[i : i + 50], repo_type="dataset"):
                if getattr(info, "lfs", None):
                    meta[info.path] = (info.lfs.sha256, info.lfs.size)
    except Exception as exc:
        print(f"  NOTE: could not fetch remote versions for {repo_id} ({exc}); will re-download unconditionally.")
        return {}
    return meta


def _fix_one(repo_id: str, hf_path: str, local_path: Path, study_dir: Path, output_dir: Path, remote) -> tuple[str, bool, str]:
    """Remove the extracted study folder, then download the zip unless the local copy already matches the repo.

    ``remote`` is (sha256, size) or None. Returns (hf_path, success, message).
    """
    if study_dir.exists():
        shutil.rmtree(study_dir)

    if local_path.exists():
        if remote is not None and local_path.stat().st_size == remote[1] and _sha256(local_path) == remote[0]:
            return hf_path, True, "up-to-date"
        local_path.unlink()

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        return hf_path, True, "downloaded"
    except EntryNotFoundError:
        return hf_path, False, "not found in repo"
    except Exception as exc:
        return hf_path, False, str(exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_fixed_reg_studies.py",
        description=(
            "Re-download fixed registration studies (re-uploaded to correct truncated / "
            "empty / corrupt archives) from MR-RATE HuggingFace repos. Zips are re-fetched "
            "only when the local copy is missing or does not match the repo; extracted "
            "study folders are always removed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-path",
        required=True,
        metavar="PATH",
        help="Path to the JSON manifest of fixed studies (nested by derivative).",
    )
    parser.add_argument(
        "--output-base",
        required=True,
        metavar="DIR",
        help="Root data directory (same as --output-base in download.py).",
    )
    parser.add_argument(
        "--coreg",
        action="store_true",
        default=False,
        help="Re-download fixed coreg-space studies from Forithmus/MR-RATE-coreg. (default: disabled)",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        default=False,
        help="Re-download fixed atlas-space studies from Forithmus/MR-RATE-atlas. (default: disabled)",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        metavar="N",
        help="Concurrent download threads. (default: 8)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.coreg and not args.atlas:
        print("ERROR: At least one of --coreg or --atlas must be specified.")
        parser.print_usage()
        return 1

    json_path = Path(args.json_path).resolve()
    data_root = Path(args.output_base).resolve()

    if not json_path.exists():
        print(f"ERROR: JSON manifest not found: {json_path}")
        return 1

    with open(json_path) as f:
        manifest: dict = json.load(f)

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    api = HfApi()

    active_derivatives = {k: v for k, v in MRI_DERIVATIVES.items() if getattr(args, k)}

    # Build the flat list of tasks. The manifest is nested derivative -> batch -> [study_uids];
    # each derivative has its own set of affected studies (coreg and atlas are not symmetric).
    tasks = []
    paths_by_repo = {}
    for deriv_key, (repo_id, zip_suffix, out_subdir) in active_derivatives.items():
        batches = manifest.get(deriv_key)
        if not batches:
            print(f"  NOTE: manifest has no '{deriv_key}' entry; skipping.")
            continue
        output_dir = data_root / out_subdir
        for batch_id, study_uids in batches.items():
            for uid in study_uids:
                zip_name = f"{uid}{zip_suffix}.zip"
                hf_path = f"mri/{batch_id}/{zip_name}"
                batch_dir = output_dir / "mri" / batch_id
                tasks.append((repo_id, hf_path, batch_dir / zip_name, batch_dir / uid, output_dir))
                paths_by_repo.setdefault(repo_id, []).append(hf_path)

    if not tasks:
        print("No studies to download.")
        return 0

    # Create output dirs
    output_dirs = {t[4] for t in tasks}
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("MR-RATE Fixed Registration Studies Downloader")
    print("=" * 60)
    print(f"  Manifest      : {json_path}")
    print(f"  Output base   : {data_root}")
    print(f"  Derivatives   : {', '.join(active_derivatives)}")
    print(f"  Studies       : {len(tasks)}")
    print(f"  Workers (DL)  : {args.download_workers}")
    print()

    # Fetch remote LFS hashes so local zips that already match can be skipped.
    print("  Checking remote versions ...")
    remote_meta = {}
    for repo_id, hf_paths in paths_by_repo.items():
        for path, meta in _remote_meta(api, repo_id, hf_paths).items():
            remote_meta[(repo_id, path)] = meta
    print()

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {
            executor.submit(
                _fix_one, repo_id, hf_path, local_path, study_dir, output_dir, remote_meta.get((repo_id, hf_path))
            ): hf_path
            for repo_id, hf_path, local_path, study_dir, output_dir in tasks
        }

        bar = tqdm(as_completed(futures), total=len(tasks), unit="study", desc="Fixing")
        for future in bar:
            hf_path, success, msg = future.result()
            if not success:
                n_failed += 1
                bar.write(f"  ERROR [{hf_path}]: {msg}")
            elif msg == "up-to-date":
                n_skipped += 1
            else:
                n_downloaded += 1
            bar.set_postfix(ok=n_downloaded, skip=n_skipped, fail=n_failed)
        bar.close()

    print()
    print(f"  Downloaded : {n_downloaded}")
    print(f"  Skipped    : {n_skipped} (local zip already matches the repo)")
    print(f"  Failed     : {n_failed}")
    print(f"  Note       : extracted study folders were removed; unzip the corrected zips next.")
    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
