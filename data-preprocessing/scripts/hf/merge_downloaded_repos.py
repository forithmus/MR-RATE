"""
MR-RATE Repo Merger
===================
Merges extracted study folders from derivative repos (coreg, atlas, nvseg-ctmr)
into the base MR-RATE repo by rsync-ing each batch directory in-place.

Mirrors the interface of download.py: same --output-base, same modality flags.
Repo directories are resolved by their known names under --output-base, so
unrelated folders in that directory are never touched.

Filenames across repos are non-colliding by design.

Arguments
---------
  --coreg  / --no-coreg       Merge co-registered MRI (MR-RATE-coreg/).    (default: disabled)
  --atlas  / --no-atlas       Merge atlas-space MRI (MR-RATE-atlas/).       (default: disabled)
  --nvseg / --no-nvseg        Merge NV-Segment-CTMR segmentations (MR-RATE-nvseg-ctmr/). (default: disabled)
  --batches BATCHES           Batches to merge. 'all' or comma-separated
                              zero-padded numbers, e.g. '00,02,10'.         (default: all)
  --output-base DIR           Root directory produced by download.py.       (default: ./data)

Usage examples
--------------
    # Merge coreg and atlas into native for all batches
    python merge_downloaded_repos.py --coreg --atlas

    # Only batches 00 and 01, custom output base
    python merge_downloaded_repos.py --coreg --batches 00,01 --output-base /data

    # All derivatives
    python merge_downloaded_repos.py --coreg --atlas --nvseg
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


KNOWN_BATCHES: List[str] = [f"batch{str(i).zfill(2)}" for i in range(28)]

DEFAULT_OUTPUT_BASE = Path("./data")

BASE_REPO_NAME = "MR-RATE"

# (flag_dest, repo_dir_name) — order defines merge order
DERIVATIVE_REPOS: List[Tuple[str, str]] = [
    ("coreg",     "MR-RATE-coreg"),
    ("atlas",     "MR-RATE-atlas"),
    ("nvseg",      "MR-RATE-nvseg-ctmr"),
]


def _mv(src: Path, dst: Path) -> None:
    """Rename src → dst, falling back to shutil.move across filesystems."""
    try:
        src.rename(dst)
    except OSError:
        shutil.move(str(src), str(dst))


def _merge_batch(src_batch: Path, dst_batch: Path) -> None:
    """
    Move all study subdirs from src_batch into dst_batch using rename.

    For each study, subdirs that don't exist in dst are moved wholesale
    (instant rename). Subdirs that already exist (e.g. transform/ contributed
    by multiple repos) are merged file-by-file.
    """
    for src_study in sorted(src_batch.iterdir()):
        if not src_study.is_dir():
            continue
        dst_study = dst_batch / src_study.name
        dst_study.mkdir(exist_ok=True)
        for src_subdir in src_study.iterdir():
            dst_subdir = dst_study / src_subdir.name
            if not dst_subdir.exists():
                _mv(src_subdir, dst_subdir)
            else:
                # Destination subdir already exists — merge file-by-file
                for src_file in src_subdir.iterdir():
                    _mv(src_file, dst_subdir / src_file.name)
                src_subdir.rmdir()
        src_study.rmdir()


def _normalise_batch_id(raw: str) -> str:
    raw = raw.strip()
    return raw if raw.startswith("batch") else f"batch{raw.zfill(2)}"


def _resolve_batches(batches_arg: str) -> List[str]:
    if batches_arg.strip().lower() == "all":
        return list(KNOWN_BATCHES)
    requested = [_normalise_batch_id(b) for b in batches_arg.split(",")]
    unknown = [b for b in requested if b not in KNOWN_BATCHES]
    if unknown:
        print(f"WARNING: Unknown batches will be skipped: {', '.join(unknown)}")
    resolved = [b for b in requested if b in KNOWN_BATCHES]
    if not resolved:
        print("ERROR: No valid batches specified.")
        sys.exit(1)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge_downloaded_repos.py",
        description=(
            "Merge derivative MR-RATE repos into MR-RATE/ using rsync. "
            "Repo directories are resolved by their known names under --output-base."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Merge coreg and atlas for all batches
  python merge_downloaded_repos.py --coreg --atlas

  # Merge all derivatives for batches 00 and 01
  python merge_downloaded_repos.py --coreg --atlas --nvseg --batches 00,01

  # Custom output base
  python merge_downloaded_repos.py --coreg --output-base /data
""",
    )

    modality_group = parser.add_argument_group("derivative modalities to merge")
    modality_group.add_argument(
        "--coreg",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Merge MR-RATE-coreg/ into MR-RATE/. (default: disabled)",
    )
    modality_group.add_argument(
        "--atlas",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Merge MR-RATE-atlas/ into MR-RATE/. (default: disabled)",
    )
    modality_group.add_argument(
        "--nvseg",
        default=False,
        action=argparse.BooleanOptionalAction,
        dest="nvseg",
        help="Merge MR-RATE-nvseg-ctmr/ into MR-RATE/. (default: disabled)",
    )

    parser.add_argument(
        "--batches",
        default="all",
        metavar="BATCHES",
        help=(
            "Batches to merge. Use 'all' or a comma-separated list of "
            "zero-padded numbers, e.g. '00,02,10'. (default: all)"
        ),
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        metavar="DIR",
        help=(
            "Root directory produced by download.py. MR-RATE/ and derivative "
            "repo directories must live here. (default: ./data)"
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    output_base = Path(args.output_base).resolve()

    # Validate base repo
    base_repo = output_base / BASE_REPO_NAME
    if not base_repo.is_dir():
        print(f"ERROR: Base repo '{BASE_REPO_NAME}/' not found under {output_base}")
        sys.exit(1)

    # Resolve selected derivative repos, checking names explicitly
    selected_derivatives: List[Path] = []
    for flag_dest, repo_name in DERIVATIVE_REPOS:
        if not getattr(args, flag_dest):
            continue
        repo_path = output_base / repo_name
        if not repo_path.is_dir():
            print(f"ERROR: '{repo_name}/' not found under {output_base}")
            sys.exit(1)
        selected_derivatives.append(repo_path)

    if not selected_derivatives:
        print("ERROR: No derivative modalities selected. Pass at least one of --coreg, --atlas, --nvseg.")
        sys.exit(1)

    selected_batches = _resolve_batches(args.batches)

    print()
    print("=" * 60)
    print("MR-RATE Repo Merger")
    print("=" * 60)
    print(f"  Output base : {output_base}")
    print(f"  Base repo   : {BASE_REPO_NAME}/")
    print(f"  Derivatives : {', '.join(r.name for r in selected_derivatives)}")
    print(f"  Batches     : {', '.join(selected_batches)}")
    print()

    for batch_id in selected_batches:
        for repo in selected_derivatives:
            src = repo / "mri" / batch_id
            if not src.is_dir():
                print(f"  [{batch_id}] {repo.name}: mri/{batch_id}/ not found, skipping.")
                continue

            print(f"  [{batch_id}] {repo.name} → {BASE_REPO_NAME} ... ", end="", flush=True)
            try:
                dst = base_repo / "mri" / batch_id
                dst.mkdir(parents=True, exist_ok=True)
                _merge_batch(src, dst)
                src.rmdir()  # remove now-empty batch dir
                print("done")
            except Exception as exc:
                print("FAILED")
                print(f"  {exc}")

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())