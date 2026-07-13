# MR-RATE Fixed Registration Studies

Some studies in the
[`Forithmus/MR-RATE-coreg`](https://huggingface.co/datasets/Forithmus/MR-RATE-coreg) and
[`Forithmus/MR-RATE-atlas`](https://huggingface.co/datasets/Forithmus/MR-RATE-atlas)
repositories were re-processed to correct data defects in the registration outputs. In both
issues below, the affected studies were re-registered from the native-space MRI in
[`Forithmus/MR-RATE`](https://huggingface.co/datasets/Forithmus/MR-RATE) and the corrected
coreg/atlas zips were re-uploaded, **overwriting** the previous ones on `main` (they do not
add new studies). This page explains how to pick up the corrected files depending on how you
originally downloaded the dataset.

> **If you have a local copy, re-running `download.py` alone will NOT fix these studies.**
> The affected zips already exist locally, so a normal re-run skips them. You must remove the
> stale local copies first (Options 2-3 below handle this for you).

## What was fixed?

Two defects, each confined to a set of batches. The affected `study_uid`s are listed per
issue, grouped by derivative and batch, in these manifests:

- [`scripts/hf/fixed_truncated_reg_study_ids.json`](../scripts/hf/fixed_truncated_reg_study_ids.json):
  **truncated `.nii.gz` outputs** (`EOFError: Compressed file ended before the
  end-of-stream marker`) in **batch03-05**, caused by registration jobs interrupted
  mid-write. 130 coreg (36GB) + 152 atlas (20GB) studies re-uploaded on 2026-06-03.
- [`scripts/hf/fixed_corrupt_reg_study_ids.json`](../scripts/hf/fixed_corrupt_reg_study_ids.json):
  **corrupted archives** in **batch06 / batch08 / batch09**
  with two distinct defects ([reported here](https://huggingface.co/datasets/Forithmus/MR-RATE-coreg/discussions/1)):
  - a `*_coreg.zip` (or `*_atlas.zip`) is an empty 22-byte archive (0 entries), so the whole
    study is missing; and
  - a single `.nii.gz` inside an otherwise-valid zip is truncated / zero-byte (fails gzip
    decompression).

  The report only flagged coreg, but since atlas comes from the same pipeline run and is very
  likely affected too, the same studies were re-registered and re-uploaded for atlas as a
  precaution. 70 studies re-uploaded in each of coreg (27GB) and atlas (9GB) on 2026-07-13.

The coreg and atlas repos are **not necessarily symmetric** (a study fixed in coreg may not be fixed in
atlas), so each manifest lists the two derivatives separately:

```json
{
    "coreg": {"batch03": ["uid1", ...], ...},
    "atlas": {"batch03": ["uid1", ...], ...}
}
```

## How to get the corrected studies

### Option 1: You haven't downloaded the dataset yet

Follow the [Downloading Dataset](../README.md#downloading-dataset) instructions. The
corrected studies are already included in the full repository downloads.

### Option 2: You downloaded with git LFS

Pull from the remote and the overwritten zips will be fetched automatically:

```bash
git -C <local-MR-RATE-coreg-repo> pull
git -C <local-MR-RATE-atlas-repo> pull
```

If you had already unzipped, delete the affected extracted study folders before re-unzipping
so the corrected files replace the stale ones, then unzip with the `find`/`unzip` block from
Option 3.

### Option 3: You downloaded with `download.py` or `snapshot_download`

Use the dedicated fix script together with each manifest. For each listed study it removes the
extracted study folder and re-downloads the zip **only if** the local copy is missing or does
not match the repo (verified by LFS SHA-256):

```bash
python scripts/hf/download_fixed_reg_studies.py \
    --json-path scripts/hf/fixed_truncated_reg_study_ids.json \
    --output-base ./data \
    --coreg --atlas \
    --download-workers 8

python scripts/hf/download_fixed_reg_studies.py \
    --json-path scripts/hf/fixed_corrupt_reg_study_ids.json \
    --output-base ./data \
    --coreg --atlas \
    --download-workers 8
```

See `python scripts/hf/download_fixed_reg_studies.py --help` for the full details of options.

> **Safe to resume, but it deletes files permanently.** The script is resumable: on each run
> it checks every listed zip against the repo and skips the ones that already match, so
> interrupting and re-running only fetches what is still missing or stale. However, the
> deletions are **permanent and unrecoverable**: for every listed study it removes the
> extracted study folder (`mri/<batch>/<uid>/`) on every run, and deletes any local zip that
> does not match the repo before re-downloading. Extracted folders are always removed because
> unzipping is a separate follow-up step (below) and a partial extraction cannot be trusted.
> Only the studies in the manifest are touched; the rest of your local dataset is left
> untouched.

Once the download completes, unzip in parallel and delete the zips to reclaim disk space
(adjust `-P 4` to match your CPU count; the extracted folders were already removed, so files
re-extract fresh):

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