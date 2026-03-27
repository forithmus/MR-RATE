"""
MR-RATE HuggingFace Dataset Downloader
=======================================
Downloads MRI zips, metadata CSVs, and/or report CSVs from the gated
MR-RATE dataset family on HuggingFace, with optional on-the-fly unzipping.

HuggingFace repositories
-------------------------
    Forithmus/MR-RATE          — native-space MRI + metadata + reports
    Forithmus/MR-RATE-coreg    — co-registered MRI  (_coreg.zip)
    Forithmus/MR-RATE-atlas    — atlas-space MRI     (_atlas.zip)
    Forithmus/MR-RATE-vista-seg — VISTA segmentations (_vista-seg.zip)

Repository layout (all repos share the same mri/ structure)
------------------------------------------------------------
    mri/
        batch00/
            <study_uid>[<suffix>].zip
        batch01/ ...
    metadata/               ← Forithmus/MR-RATE only
        batch00_metadata.csv ...
    reports/                ← Forithmus/MR-RATE only
        batch00_reports.csv ...

Local output layout
-------------------
Each repo is downloaded into its own subdirectory under --output-base
(default: ./data) so that HuggingFace caches and zip files never mix:

    ./data/
        MR-RATE/
            mri/batch00/<study_uid>.zip
            mri/batch00/<study_uid>/   ← when --unzip
            metadata/batch00_metadata.csv
            reports/batch00_reports.csv
        MR-RATE-coreg/
            mri/batch00/<study_uid>_coreg.zip
            mri/batch00/<study_uid>/   ← when --unzip
        MR-RATE-atlas/
            mri/batch00/<study_uid>_atlas.zip
            mri/batch00/<study_uid>/   ← when --unzip
        MR-RATE-vista-seg/
            mri/batch00/<study_uid>_vista-seg.zip
            mri/batch00/<study_uid>/   ← when --unzip

How it works
------------
The script operates batch by batch. For each selected batch it:
  1. Downloads metadata and/or reports (single-file downloads, native repo only).
  2. For each enabled MRI derivative, downloads all zips for that batch via
     snapshot_download (concurrent, resumable across interrupted runs).
  3. Optionally unzips the downloaded studies in parallel and optionally removes
     the source zips to reclaim disk space.
  4. Prints a download status table after every run, showing per-batch completion
     across all repos.

Note 1: If unzipping is interrupted, there is no mechanism to check if content
of a study folder is complete.

Note 2: snapshot_download ignores zip files that are already present in the local
directory by default using output_dir/.cache but if you delete zips after extraction,
huggingface_hub will download them again. One solution is to pass ignore_patterns for 
existing study_uid folders but resolving the file tree on server side takes too long 
due excessive number of files in the repos. So we ignored this for now.

Note 3: Inside each locally downloaded repo directory, there are HF .cache folders 
produced by snapshot_download, if your file system has number of files limit, you 
might want to delete them.

Dependencies
------------
    pip install huggingface_hub tqdm

Authentication
--------------
This dataset is gated. You must authenticate before running:
    huggingface-cli login
    # or export HF_TOKEN=<your_token>

Arguments
---------
Content selection:
  --batches BATCHES           Batches to download. 'all' or comma-separated
                              zero-padded numbers, e.g. '00,02,10'. (default: all)
  --native / --no-native      Download native-space MRI. (default: enabled)
  --coreg  / --no-coreg       Download co-registered MRI. (default: disabled)
  --atlas  / --no-atlas       Download atlas-space MRI. (default: disabled)
  --vista-seg / --no-vista-seg Download VISTA segmentations. (default: disabled)
  --no-mri                    Disable all MRI derivatives (overrides the above).
  --metadata / --no-metadata  Download metadata CSV files. (default: enabled)
  --reports  / --no-reports   Download report CSV. (default: enabled)

Post-download:
  --unzip                     Extract each batch's zips after downloading. (default: disabled)
  --delete-zips               Delete zip files after extraction. Requires --unzip. (default: disabled)

Performance:
  --download-workers N        Concurrent workers for snapshot_download. (default: 8)
  --unzip-workers N           Parallel workers for unzipping. Requires --unzip. (default: 4)
  --timeout SECONDS           HuggingFace download timeout. (default: 300)
  --xet-high-perf             Set HF_XET_HIGH_PERFORMANCE=1 to enable the
                              high-performance Xet transfer backend.
                              WARNING: overrides --download-workers, uses all
                              available CPUs and maximum bandwidth — system
                              responsiveness may degrade. (default: disabled)

Output:
  --output-base DIR           Root directory; each repo gets its own subdirectory.
                              (default: ./data)

Post-download status:
  A download status table is always printed after every run. It queries the
  remote repositories and compares with local files, showing for each batch 
  and repository whether data is fully downloaded (✅), complete but mixed 
  zip/folder state (☑️), partially downloaded (🔄), missing (❌), or whether 
  the remote listing could not be fetched (⚠️).

Usage examples
--------------
    # Native MRI only for all batches, unzip and free disk as you go (use xet-high-perf for faster downloads)
    python download.py --batches all --unzip --delete-zips --no-metadata --no-reports --xet-high-perf

    # Download coreg and atlas for batches 00 and 01, no native, no metadata/reports
    python download.py --batches 00,01 --no-native --coreg --atlas --no-metadata --no-reports

    # All derivatives, keep zips, custom output base
    python download.py --native --coreg --atlas --vista-seg --no-metadata --no-reports \\
        --output-base /data

    # Metadata and reports only (no MRI)
    python download.py --no-mri

    # Check download status without downloading anything
    python download.py --no-mri --no-metadata --no-reports
"""

import argparse
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple


REPO_TYPE = "dataset"

# (hf_repo_id, zip_suffix, output_subdir_name)
MRI_DERIVATIVES = {
    "native":    ("Forithmus/MR-RATE",           "",           "MR-RATE"),
    "coreg":     ("Forithmus/MR-RATE-coreg",     "_coreg",     "MR-RATE-coreg"),
    "atlas":     ("Forithmus/MR-RATE-atlas",     "_atlas",     "MR-RATE-atlas"),
    "vista_seg": ("Forithmus/MR-RATE-vista-seg", "_vista-seg", "MR-RATE-vista-seg"),
}

NATIVE_REPO_ID = MRI_DERIVATIVES["native"][0]   # used for metadata / reports
NATIVE_OUTPUT_SUBDIR = MRI_DERIVATIVES["native"][2]

# Batches are fixed: batch00 … batch27
KNOWN_BATCHES: List[str] = [f"batch{str(i).zfill(2)}" for i in range(28)]

DEFAULT_OUTPUT_BASE = Path("./data")
SNAPSHOT_DOWNLOAD_MAX_RETRIES = 5
SNAPSHOT_DOWNLOAD_RETRY_DELAY = 5


def _hf_imports():
    """Lazy import after env vars are set."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )
        sys.exit(1)
    return hf_hub_download, snapshot_download, EntryNotFoundError, list_repo_files


def _require_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        print(
            "ERROR: tqdm is not installed.\n"
            "Install it with:  pip install tqdm"
        )
        sys.exit(1)


def _normalise_batch_id(raw: str) -> str:
    """Turn '0', '00', '2', 'batch02' → 'batch00', 'batch02', etc."""
    raw = raw.strip()
    if raw.startswith("batch"):
        return raw
    return f"batch{raw.zfill(2)}"


def _resolve_batches(batches_arg: str) -> List[str]:
    if batches_arg.strip().lower() == "all":
        return list(KNOWN_BATCHES)

    requested = [_normalise_batch_id(b) for b in batches_arg.split(",")]
    missing = [b for b in requested if b not in KNOWN_BATCHES]
    if missing:
        print(
            f"WARNING: The following batches are not in the known range "
            f"(batch00–batch{KNOWN_BATCHES[-1]}) and will be skipped: "
            f"{', '.join(missing)}"
        )
    resolved = [b for b in requested if b in KNOWN_BATCHES]
    if not resolved:
        print("ERROR: None of the requested batches are valid.")
        sys.exit(1)
    return resolved


def _download_file(hf_hub_download, repo_id: str, filename: str, output_dir: Path) -> Path:
    """Download a single HF file into output_dir, preserving the repo path."""
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(output_dir),
    )
    return Path(local_path)


def _unzip_worker(args: Tuple[str, bool]) -> Tuple[str, bool, str]:
    """
    Top-level worker for multiprocessing.Pool: extracts a zip and optionally deletes it.

    Args:
        args: (zip_path_str, delete_after)

    Returns:
        (zip_path_str, success, error_message)
    """
    zip_path_str, delete_after = args
    zip_path = Path(zip_path_str)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)
        if delete_after:
            zip_path.unlink()
        return zip_path_str, True, ""
    except Exception as exc:
        return zip_path_str, False, str(exc)


def download_metadata(hf_hub_download, batch_id: str, output_dir: Path, EntryNotFoundError) -> None:
    filename = f"metadata/{batch_id}_metadata.csv"
    local = output_dir / filename
    if local.exists():
        print(f"  [metadata] Already exists, skipping: {local}")
        return
    print(f"  [metadata] Downloading {filename} ...")
    try:
        _download_file(hf_hub_download, NATIVE_REPO_ID, filename, output_dir)
        print(f"  [metadata] Saved to {local}")
    except EntryNotFoundError:
        print(f"  [metadata] WARNING: Not found in repo: {filename}")
    except Exception as exc:
        print(f"  [metadata] ERROR downloading {filename}: {exc}")


def download_reports(hf_hub_download, batch_id: str, output_dir: Path, EntryNotFoundError) -> None:
    filename = f"reports/{batch_id}_reports.csv"
    local = output_dir / filename
    if local.exists():
        print(f"  [reports]  Already exists, skipping: {local}")
        return
    print(f"  [reports]  Downloading {filename} ...")
    try:
        _download_file(hf_hub_download, NATIVE_REPO_ID, filename, output_dir)
        print(f"  [reports]  Saved to {local}")
    except EntryNotFoundError:
        print(f"  [reports]  WARNING: Not found in repo: {filename}")
    except Exception as exc:
        print(f"  [reports]  ERROR downloading {filename}: {exc}")


def download_mri(
    snapshot_download,
    batch_id: str,
    output_dir: Path,
    repo_id: str,
    zip_suffix: str,
    unzip: bool,
    delete_zips: bool,
    download_workers: int,
    unzip_workers: int,
    tqdm,
) -> None:
    label = f"[mri/{zip_suffix or 'native'}]".ljust(20)
    batch_dir = output_dir / "mri" / batch_id

    # Phase 1 — bulk download via snapshot_download (concurrent, resumable)
    print(f"  {label} Downloading {batch_id} from {repo_id} with {download_workers} workers ...")
    for attempt in range(1, SNAPSHOT_DOWNLOAD_MAX_RETRIES + 1):
        try:
            response = snapshot_download(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                allow_patterns=f"mri/{batch_id}/*.zip",
                local_dir=str(output_dir),
                max_workers=download_workers,
            )
            if Path(response).resolve() != output_dir.resolve():
                print(
                    f"  {label} WARNING: snapshot_download returned unexpected path "
                    f"{response!r} (expected {output_dir}), retrying in {SNAPSHOT_DOWNLOAD_RETRY_DELAY}s..."
                )
                time.sleep(SNAPSHOT_DOWNLOAD_RETRY_DELAY)
                continue
            break
        except Exception as exc:
            if attempt < SNAPSHOT_DOWNLOAD_MAX_RETRIES:
                print(
                    f"  {label} ERROR during snapshot_download for {batch_id} "
                    f"(attempt {attempt}/{SNAPSHOT_DOWNLOAD_MAX_RETRIES}): {exc}\n"
                    f"  {label} Retrying in {SNAPSHOT_DOWNLOAD_RETRY_DELAY}s..."
                )
                time.sleep(SNAPSHOT_DOWNLOAD_RETRY_DELAY)
            else:
                print(
                    f"  {label} ERROR during snapshot_download for {batch_id} "
                    f"(attempt {attempt}/{SNAPSHOT_DOWNLOAD_MAX_RETRIES}): {exc}\n"
                    f"  {label} All {SNAPSHOT_DOWNLOAD_MAX_RETRIES} attempts exhausted. Skipping."
                )
                return

    if not unzip:
        n_zips = len(list(batch_dir.glob("*.zip"))) if batch_dir.exists() else 0
        print(f"  {label} {batch_id}: {n_zips} zip(s) downloaded.")
        return

    # Phase 2 — parallel unzip
    all_zips = sorted(batch_dir.glob("*.zip")) if batch_dir.exists() else []
    if not all_zips:
        print(f"  {label} WARNING: No zip files found in {batch_dir} after download.")
        return

    # The zip internal root is always <study_uid>/ regardless of the zip filename suffix,
    # so we strip the suffix from the stem to find the actual extracted directory.
    to_unzip = []
    n_already = 0
    for zip_path in all_zips:
        stem = zip_path.stem
        study_dir = zip_path.parent / (stem[: -len(zip_suffix)] if zip_suffix else stem)
        if study_dir.exists():
            n_already += 1
            if delete_zips:
                zip_path.unlink(missing_ok=True)
        else:
            to_unzip.append(zip_path)

    if n_already:
        print(f"  {label} {batch_id}: {n_already} already extracted (skipped).")

    if not to_unzip:
        return

    worker_args = [(str(zp), delete_zips) for zp in to_unzip]
    n_failed = 0

    print(
        f"  {label} Extracting {len(to_unzip)} studies "
        f"with {unzip_workers} worker(s) ..."
    )
    with Pool(processes=unzip_workers) as pool:
        bar = tqdm(
            pool.imap_unordered(_unzip_worker, worker_args),
            total=len(to_unzip),
            desc=f"  {label} extracting {batch_id}",
            unit="study",
            leave=True,
        )
        for zip_path_str, success, error_msg in bar:
            if not success:
                n_failed += 1
                bar.write(f"    ERROR extracting {Path(zip_path_str).name}: {error_msg}")
        bar.close()

    action = "extracted and deleted zips" if delete_zips else "extracted"
    print(
        f"  {label} {batch_id}: {len(to_unzip) - n_failed}/{len(to_unzip)} studies "
        f"{action}{f', {n_failed} failed' if n_failed else ''}."
    )


def print_download_status(list_repo_files, batches: List[str], output_base: Path) -> None:
    """
    Print a per-batch download status table covering all MRI derivatives plus
    metadata and reports, comparing remote HuggingFace file listings against
    what is present locally (as zips and/or extracted study folders).

    Emoji legend
    ------------
    ✅  All studies present and consistently in one form (all zips or all folders).
    ☑️  All studies present but mixed: some still zipped, some already extracted.
    🔄  Partially downloaded (some studies missing).
    ❌  Nothing found locally for this batch.
    ⚠️  Could not retrieve the remote file list (network/auth error).
    """
    # Helper functions for visual width calculation
    def _visual_width(s: str) -> int:
        """
        Return the terminal display width of s.

        Wide emojis count as 2 columns. Variation Selector-16
        (U+FE0F) forces the preceding character into emoji presentation (also 2 wide)
        and itself contributes 1 extra column on top of the base character's 1.
        Zero-width combining marks contribute 0.
        """
        import unicodedata
        width = 0
        for ch in s:
            cp = ord(ch)
            if cp == 0xFE0F:
                # VS-16: upgrades preceding char from 1→2; add the extra column here
                width += 1
            elif unicodedata.category(ch) in ("Mn", "Cf"):
                pass  # zero-width combining / format chars
            elif unicodedata.east_asian_width(ch) in ("W", "F"):
                width += 2
            else:
                width += 1
        return width

    def _ljust_vis(s: str, width: int) -> str:
        """Left-justify s to the given visual (terminal) width."""
        return s + " " * max(0, width - _visual_width(s))

    def _center_vis(s: str, width: int) -> str:
        """Centre s within the given visual (terminal) width."""
        pad = max(0, width - _visual_width(s))
        left = pad // 2
        return " " * left + s + " " * (pad - left)

    # Step 1 — Fetch remote file listings once per repo
    print("\nFetching remote file listings for status check ...")

    # repo_files: repo_id → frozenset of HF-relative path strings
    repo_files: dict = {}
    for _label, (repo_id, _suffix, _subdir) in MRI_DERIVATIVES.items():
        if repo_id not in repo_files:
            try:
                repo_files[repo_id] = frozenset(
                    list_repo_files(repo_id, repo_type=REPO_TYPE)
                )
            except Exception as exc:
                print(f"  WARNING: could not list {repo_id}: {exc}")
                repo_files[repo_id] = None  # sentinel → show "⚠️" in table

    # Native repo also carries metadata and reports
    native_files = repo_files.get(NATIVE_REPO_ID)

    # Step 2 — Build the table rows
    # Columns: batch | native | coreg | atlas | vista_seg | metadata | reports
    DERIV_ORDER = list(MRI_DERIVATIVES.keys())  # preserves insertion order

    # We want to track whether any batch has a mixed-zip footnote
    mixed_footnote_needed = False

    rows = []
    for batch_id in batches:
        row: dict = {"batch": batch_id}

        # -- MRI derivatives --
        for label, (repo_id, zip_suffix, out_subdir) in MRI_DERIVATIVES.items():
            hf_set = repo_files.get(repo_id)
            if hf_set is None:
                row[label] = ("⚠️", "")
                continue

            # Files in this batch on HF
            prefix = f"mri/{batch_id}/"
            hf_zips = [
                f for f in hf_set
                if f.startswith(prefix) and f.endswith(".zip")
            ]
            total = len(hf_zips)
            if total == 0:
                row[label] = ("⚠️", "")
                continue

            batch_local_dir = output_base / out_subdir / "mri" / batch_id
            n_zip = 0
            n_folder = 0
            n_downloaded = 0
            for hf_path in hf_zips:
                zip_name = Path(hf_path).name          # e.g. "uid_coreg.zip"
                zip_stem = Path(hf_path).stem           # e.g. "uid_coreg"
                # Strip derivative suffix to get the study uid
                if zip_suffix and zip_stem.endswith(zip_suffix):
                    study_uid = zip_stem[: -len(zip_suffix)]
                else:
                    study_uid = zip_stem

                local_zip = batch_local_dir / zip_name
                local_folder = batch_local_dir / study_uid

                has_zip = local_zip.exists()
                has_folder = local_folder.is_dir()
                has_downloaded = has_zip or has_folder

                if has_zip:
                    n_zip += 1
                if has_folder:
                    n_folder += 1
                if has_downloaded:
                    n_downloaded += 1

            fraction = f"{n_downloaded}/{total}"

            if n_downloaded == 0:
                icon = "❌"
                note = ""
            elif n_downloaded < total:
                icon = "🔄"
                note = ""
            else:
                # All present — check for mixed state
                if n_zip > 0 and n_folder > 0:
                    icon = "☑️"
                    note = "*"
                    mixed_footnote_needed = True
                else:
                    icon = "✅"
                    note = ""

            row[label] = (icon, fraction + note)

        # -- Metadata --
        meta_hf_path = f"metadata/{batch_id}_metadata.csv"
        meta_local = output_base / NATIVE_OUTPUT_SUBDIR / meta_hf_path
        if native_files is None:
            row["metadata"] = "⚠️"
        elif meta_hf_path not in native_files:
            row["metadata"] = "N/A"
        else:
            row["metadata"] = "✅" if meta_local.exists() else "❌"

        # -- Reports --
        rep_hf_path = f"reports/{batch_id}_reports.csv"
        rep_local = output_base / NATIVE_OUTPUT_SUBDIR / rep_hf_path
        if native_files is None:
            row["reports"] = "⚠️"
        elif rep_hf_path not in native_files:
            row["reports"] = "N/A"
        else:
            row["reports"] = "✅" if rep_local.exists() else "❌"

        rows.append(row)

    # Step 3 — Render the table
    # MRI cell content: "<icon>  <fraction>" — fixed width for alignment
    MRI_CELL_W = 14  # enough for "✔   120/120*"
    META_CELL_W = 8

    deriv_labels = {
        "native":    "native",
        "coreg":     "coreg",
        "atlas":     "atlas",
        "vista_seg": "vista-seg",
    }

    header_parts = [f"{'Batch':<10}"]
    for label in DERIV_ORDER:
        header_parts.append(f" {deriv_labels[label]:^{MRI_CELL_W}}")
    header_parts.append(f" {'metadata':^{META_CELL_W}}")
    header_parts.append(f" {'reports':^{META_CELL_W}}")

    col_sep = "│"
    header = col_sep.join(header_parts)
    divider = "─" * 10 + "┼" + ("─" * (MRI_CELL_W + 1) + "┼") * len(DERIV_ORDER) + \
              ("─" * (META_CELL_W + 1) + "┼") + "─" * (META_CELL_W + 1)

    print()
    print("Download Status")
    print("=" * len(header))
    print(header)
    print(divider)

    for row in rows:
        parts = [f"{row['batch']:<10}"]
        for label in DERIV_ORDER:
            icon, fraction = row[label]
            cell = f"{icon}  {fraction}" if fraction else icon
            parts.append(" " + _ljust_vis(cell, MRI_CELL_W))
        parts.append(" " + _center_vis(row["metadata"], META_CELL_W))
        parts.append(" " + _center_vis(row["reports"], META_CELL_W))
        print(col_sep.join(parts))

    print("=" * len(header))

    if mixed_footnote_needed:
        print("  * mixed zip/folder: all studies are present but there are both zips and extracted folders in the same batch")

    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download.py",
        description=(
            "Download MRI zips, metadata, and/or reports from the "
            "MR-RATE HuggingFace dataset family."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Native MRI for all batches, unzip and delete zips as you go
  python download.py --batches all --unzip --delete-zips --no-metadata --no-reports

  # Co-registered + atlas MRI for batches 00 and 01, no native
  python download.py --batches 00,01 --no-native --coreg --atlas --no-metadata --no-reports

  # All derivatives, keep zips, custom output base
  python download.py --native --coreg --atlas --vista-seg --no-metadata --no-reports \\
      --output-base /data

  # Metadata and reports only (no MRI at all)
  python download.py --no-mri

  # Use xet high-performance backend for faster downloads
  python download.py --native --coreg --unzip --delete-zips --no-metadata --no-reports \\
      --xet-high-perf --unzip-workers 8

  # Check download status without downloading anything
  python download.py --no-mri --no-metadata --no-reports
""",
    )

    parser.add_argument(
        "--batches",
        default="all",
        metavar="BATCHES",
        help=(
            "Batches to download. Use 'all' for every batch, or a comma-separated "
            "list of zero-padded or plain numbers, e.g. '00', '00,02,10'. (default: all)"
        ),
    )

    # MRI derivative flags
    mri_group = parser.add_argument_group("MRI derivatives")
    mri_group.add_argument(
        "--native",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Download native-space MRI from Forithmus/MR-RATE. (default: enabled)",
    )
    mri_group.add_argument(
        "--coreg",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Download co-registered MRI from Forithmus/MR-RATE-coreg. (default: disabled)",
    )
    mri_group.add_argument(
        "--atlas",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Download atlas-space MRI from Forithmus/MR-RATE-atlas. (default: disabled)",
    )
    mri_group.add_argument(
        "--vista-seg",
        default=False,
        action=argparse.BooleanOptionalAction,
        dest="vista_seg",
        help="Download VISTA segmentations from Forithmus/MR-RATE-vista-seg. (default: disabled)",
    )
    mri_group.add_argument(
        "--no-mri",
        action="store_true",
        default=False,
        dest="no_mri",
        help="Disable all MRI derivatives (overrides --native/--coreg/--atlas/--vista-seg).",
    )

    # Other content flags
    parser.add_argument(
        "--metadata",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Download metadata CSV files (native repo only). (default: enabled)",
    )
    parser.add_argument(
        "--reports",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Download report CSV files (native repo only). (default: enabled)",
    )

    # Post-download
    parser.add_argument(
        "--unzip",
        action="store_true",
        default=False,
        help="Extract each study zip after downloading a batch. (default: disabled)",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        default=False,
        help=(
            "Delete each zip file right after extraction. "
            "Only has effect when --unzip is also set. (default: disabled)"
        ),
    )

    # Output
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        metavar="DIR",
        help=(
            "Root directory for downloaded files. Each repo gets its own subdirectory "
            "(MR-RATE, MR-RATE-coreg, etc.). (default: ./data)"
        ),
    )

    # Performance
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Number of concurrent download workers passed to snapshot_download. "
            "(default: 8)"
        ),
    )
    parser.add_argument(
        "--unzip-workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel unzip workers. Only has effect when --unzip is set. (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SECONDS",
        help="HuggingFace download timeout in seconds. (default: 300)",
    )
    parser.add_argument(
        "--xet-high-perf",
        action="store_true",
        default=False,
        help=(
            "Set HF_XET_HIGH_PERFORMANCE=1 to enable the high-performance Xet "
            "transfer backend. WARNING: ignores --download-workers and uses all "
            "available CPUs and maximum bandwidth. (default: disabled)"
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --no-mri overrides all individual derivative flags
    if args.no_mri:
        args.native = args.coreg = args.atlas = args.vista_seg = False

    if args.delete_zips and not args.unzip:
        print(
            "\nWARNING: --delete-zips has no effect without --unzip. "
            "Zips will be kept."
        )

    output_base = Path(args.output_base).resolve()

    # Set HF env vars before any huggingface_hub import
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.timeout)

    if args.xet_high_perf:
        print(
            "\nWARNING: --xet-high-perf is enabled.\n"
            "  HF_XET_HIGH_PERFORMANCE=1 will be set for this session.\n"
            "  This overrides --download-workers and uses all available CPUs\n"
            "  and maximum network bandwidth. System responsiveness may degrade.\n"
        )
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    hf_hub_download, snapshot_download, EntryNotFoundError, list_repo_files = _hf_imports()
    tqdm = _require_tqdm()

    # Build list of active MRI derivatives with their per-derivative output dirs
    active_mri = [
        (label, repo_id, zip_suffix, output_base / out_subdir)
        for label, (repo_id, zip_suffix, out_subdir) in MRI_DERIVATIVES.items()
        if getattr(args, label)
    ]

    # Native output dir holds metadata and reports
    native_output_dir = output_base / NATIVE_OUTPUT_SUBDIR

    # Create required output dirs
    for _, _, _, mri_out in active_mri:
        mri_out.mkdir(parents=True, exist_ok=True)
    if args.metadata or args.reports:
        native_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve batches from the fixed known list
    selected_batches = _resolve_batches(args.batches)

    print()
    print("=" * 60)
    print("MR-RATE HuggingFace Dataset Downloader")
    print("=" * 60)
    print(f"  Output base : {output_base}")
    print(f"  Batches     : {', '.join(selected_batches)}")
    print(f"  MRI active  : {', '.join(l for l, *_ in active_mri) if active_mri else 'none'}")
    if active_mri:
        for label, repo_id, _, mri_out in active_mri:
            print(f"    {label:<12} → {repo_id}  →  {mri_out}")
    print(f"  Metadata    : {args.metadata}")
    print(f"  Reports     : {args.reports}")
    print(f"  Unzip       : {args.unzip}")
    print(f"  Delete zips : {args.delete_zips if args.unzip else 'N/A'}")
    print(f"  Workers (DL): {'N/A (xet-high-perf)' if args.xet_high_perf else args.download_workers}")
    print(f"  Workers (UZ): {args.unzip_workers if args.unzip else 'N/A'}")
    print(f"  Timeout (s) : {args.timeout}")
    print(f"  Xet high-perf: {'ON (HF_XET_HIGH_PERFORMANCE=1)' if args.xet_high_perf else 'off'}")
    print()

    for i, batch_id in enumerate(selected_batches, 1):
        print(f"[{i}/{len(selected_batches)}] Batch: {batch_id}")
        print("-" * 40)

        if args.metadata:
            download_metadata(hf_hub_download, batch_id, native_output_dir, EntryNotFoundError)

        if args.reports:
            download_reports(hf_hub_download, batch_id, native_output_dir, EntryNotFoundError)

        for label, repo_id, zip_suffix, mri_output_dir in active_mri:
            download_mri(
                snapshot_download=snapshot_download,
                batch_id=batch_id,
                output_dir=mri_output_dir,
                repo_id=repo_id,
                zip_suffix=zip_suffix,
                unzip=args.unzip,
                delete_zips=args.delete_zips,
                download_workers=args.download_workers,
                unzip_workers=args.unzip_workers,
                tqdm=tqdm,
            )

        print()

    print_download_status(list_repo_files, selected_batches, output_base)

    print("=" * 60)
    print("Done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())