# MR-RATE Backfilled Registration Studies

Some studies were initially missing from the
[`Forithmus/MR-RATE-coreg`](https://huggingface.co/datasets/Forithmus/MR-RATE-coreg) and
[`Forithmus/MR-RATE-atlas`](https://huggingface.co/datasets/Forithmus/MR-RATE-atlas)
repositories due to processing failures at the time of the original upload. These studies
have since been re-processed and uploaded. This page explains how to obtain them depending
on how you originally downloaded the dataset.

## Which studies were backfilled?

[`scripts/hf/backfilled_reg_study_ids.json`](../scripts/hf/backfilled_reg_study_ids.json)
is a manifest listing the affected `study_uid`s grouped by batch. The same study IDs apply
to both the coreg and atlas repos. These UIDs were absent from the original upload and do
not overlap with studies that were already present, so downloading them will not affect or
duplicate any existing data.

```json
{
    "batch06": ["uid1", "uid2", ...],
    ...
}
```

## How to get the backfilled studies

### Option 1: You haven't downloaded the dataset yet

Follow the [Downloading Dataset](../README.md#downloading-dataset) instructions. The backfilled studies are already included in the full
repository downloads.

### Option 2: You downloaded with `download.py` or `snapshot_download` and still have the zip files and `.cache` folder

Re-run `download.py` (or `snapshot_download` directly) as you did originally. All existing
zips are skipped and only the missing studies will be fetched, then unzipped:

```bash
python scripts/hf/download.py \
    --batches all --no-native --coreg --atlas \
    --no-metadata --no-reports \
    --unzip --delete-zips --xet-high-perf \
    --output-base ./data
```

### Option 3: You downloaded with `download.py` or `snapshot_download` but no longer have the zip files or `.cache` folder

Use the dedicated backfill script together with the manifest to download only the affected
studies:

```bash
python scripts/hf/download_backfilled_reg_studies.py \
    --json-path scripts/hf/backfilled_reg_study_ids.json \
    --output-base ./data \
    --coreg --atlas \
    --download-workers 8
```

See `python scripts/hf/download_backfilled_reg_studies.py --help` for the full details of options.
Re-running is safe: completed zips are skipped and any partially downloaded zips are detected and
re-downloaded automatically.

Once the downloads are complete, unzip in parallel and delete the zips to reclaim disk space
(adjust `-P 4` to match your CPU count; `unzip -n` skips files that already exist):

```bash
# Unzip coreg zips
find ./data/MR-RATE-coreg/mri -name "*.zip" -print0 |
xargs -0 -P 4 -I {} sh -c '
    zip="$1"
    dir=$(dirname "$zip")
    unzip -n "$zip" -d "$dir" && rm -f "$zip"
' sh {}

# Unzip atlas zips
find ./data/MR-RATE-atlas/mri -name "*.zip" -print0 |
xargs -0 -P 4 -I {} sh -c '
    zip="$1"
    dir=$(dirname "$zip")
    unzip -n "$zip" -d "$dir" && rm -f "$zip"
' sh {}
```

### Option 4: You downloaded with git LFS

Pull from the remote and the newly added zips will be fetched automatically:

```bash
git -C <local-MR-RATE-coreg-repo> pull
git -C <local-MR-RATE-atlas-repo> pull
```

If you need to unzip and clean up afterwards, use the same `find`/`unzip` command from
Option 3 above.

## Verifying your download is complete

After any of the options above, you can confirm all backfilled studies are now present
using `download.py`'s built-in status check without triggering any downloads:

```bash
python scripts/hf/download.py \
    --batches all --no-mri \
    --no-metadata --no-reports \
    --output-base ./data
```

This prints a per-batch table showing how many studies are present locally versus available
remotely. See the [Downloading Dataset](../README.md#downloading-dataset) for a full description of the status table and its legend.