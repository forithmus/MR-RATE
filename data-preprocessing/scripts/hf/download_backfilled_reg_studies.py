"""
MR-RATE Backfilled Registration Studies Downloader
===================================================
Downloads studies that were previously missing from the coreg and atlas
HuggingFace repositories and have since been newly uploaded. Takes a JSON
manifest that maps batch → [study_uids]. The same study IDs apply to both
the coreg and atlas repos.

JSON format (e.g. backfilled_reg_study_ids.json)
-------------------------------------------------
{
    "batch06": ["uid1", "uid2", ...],
    ...
}

Usage
-----
    python download_backfilled_reg_studies.py --json-path backfilled_reg_study_ids.json --output-base /data/root --coreg --atlas
    python download_backfilled_reg_studies.py --json-path backfilled_reg_study_ids.json --output-base /data/root --atlas --download-workers 16

Arguments
---------
    --json-path PATH      Path to the JSON manifest of backfilled studies.
    --output-base DIR     Root data directory (same as --output-base in download.py).
    --coreg               Download backfilled coreg-space studies to Forithmus/MR-RATE-coreg. (default: disabled)
    --atlas               Download backfilled atlas-space studies to Forithmus/MR-RATE-atlas. (default: disabled)
    --download-workers N  Concurrent download threads. (default: 8)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm


MRI_DERIVATIVES = {
    "coreg": ("Forithmus/MR-RATE-coreg", "_coreg", "MR-RATE-coreg"),
    "atlas": ("Forithmus/MR-RATE-atlas", "_atlas", "MR-RATE-atlas"),
}


def _is_valid_zip(path: Path) -> bool:
    """Check zip integrity by verifying the End of Central Directory signature (last 22 bytes)."""
    try:
        with open(path, "rb") as f:
            f.seek(-22, 2)
            return f.read(4) == b"PK\x05\x06"
    except (OSError, ValueError):
        return False


def _download_one(repo_id: str, hf_path: str, local_path: Path, output_dir: Path) -> tuple[str, bool, str]:
    """Download a single study zip. Returns (hf_path, success, message)."""
    if local_path.exists():
        if _is_valid_zip(local_path):
            return hf_path, True, "already exists"
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
        prog="download_backfilled_reg_studies.py",
        description=(
            "Download backfilled registration studies (previously missing, newly uploaded) "
            "from MR-RATE HuggingFace repos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-path",
        required=True,
        metavar="PATH",
        help="Path to the JSON manifest of backfilled studies.",
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
        help="Download backfilled coreg-space studies to Forithmus/MR-RATE-coreg. (default: disabled)",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        default=False,
        help="Download backfilled atlas-space studies to Forithmus/MR-RATE-atlas. (default: disabled)",
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

    active_derivatives = {k: v for k, v in MRI_DERIVATIVES.items() if getattr(args, k)}

    # Build the flat list of (repo_id, hf_path, local_path, output_dir) tasks.
    # The manifest is batch → [study_uids]; the same IDs are used for all active derivatives.
    tasks = []
    for deriv_key, (repo_id, zip_suffix, out_subdir) in active_derivatives.items():
        output_dir = data_root / out_subdir
        for batch_id, study_uids in manifest.items():
            for uid in study_uids:
                zip_name = f"{uid}{zip_suffix}.zip"
                hf_path = f"mri/{batch_id}/{zip_name}"
                local_path = output_dir / "mri" / batch_id / zip_name
                tasks.append((repo_id, hf_path, local_path, output_dir))

    if not tasks:
        print("No studies to download.")
        return 0

    # Create output dirs
    output_dirs = {t[3] for t in tasks}
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("MR-RATE Backfilled Registration Studies Downloader")
    print("=" * 60)
    print(f"  Manifest      : {json_path}")
    print(f"  Output base   : {data_root}")
    print(f"  Derivatives   : {', '.join(active_derivatives)}")
    print(f"  Studies       : {len(tasks)}")
    print(f"  Workers (DL)  : {args.download_workers}")
    print()

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {
            executor.submit(_download_one, repo_id, hf_path, local_path, output_dir): hf_path
            for repo_id, hf_path, local_path, output_dir in tasks
        }

        bar = tqdm(as_completed(futures), total=len(tasks), unit="study", desc="Downloading")
        for future in bar:
            hf_path, success, msg = future.result()
            if not success:
                n_failed += 1
                bar.write(f"  ERROR [{hf_path}]: {msg}")
            elif msg == "already exists":
                n_skipped += 1
            else:
                n_downloaded += 1
            bar.set_postfix(ok=n_downloaded, skip=n_skipped, fail=n_failed)
        bar.close()

    print()
    print(f"  Downloaded : {n_downloaded}")
    print(f"  Skipped    : {n_skipped} (already present)")
    print(f"  Failed     : {n_failed}")
    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())