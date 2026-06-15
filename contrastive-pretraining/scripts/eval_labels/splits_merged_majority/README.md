# splits_merged_majority — linear-probe labels (14 merged groups)

Ground-truth labels for the MR-RATE linear probe, built from a **3-model
majority vote** (Claude Opus 4.7 · GPT-5.5 · Nemotron-3 Super 120B; positive
when ≥2 of the available votes agree) over 37 pathologies, then collapsed into
the neuroradiologist's **14 clinical groups** (8 Pathophysiologie `PP_*` +
6 Bildphänotyp `BP_*`).

## Files the probe consumes

| File | Used by | Format |
|------|---------|--------|
| `mrrate_merged_labels.csv` | `extract_features.py --labels_file` | `study_uid` + 14 binary class columns |
| `splits.csv`               | `extract_features.py --splits_csv` | `…,study_uid,split` (train/val/test) |

Both match the `data_inference.MRReportDatasetInfer` loader contract verbatim:
`_load_labels` keys on `study_uid` and treats every other column as a class;
`_load_splits` filters on `row['split']`. The class count (14) is derived
automatically — nothing is hardcoded — so `label_names.json` is written by
`extract_features.py` straight from these column names.

Coverage: 97,896 studies — train 88,582 / val 3,764 / test 5,550.

The other files here are **not** needed by the probe:
`train.csv`/`val.csv`/`test.csv` (per-split convenience copies),
`pathologies.json` (zero-shot prompts, for `inference.py`),
`group_definitions.json` (group → member pathologies, documentation).

## Run (two steps)

```bash
cd contrastive-pretraining
LAB=scripts/eval_labels/splits_merged_majority

# 1) cache frozen-encoder features once per split
for SPLIT in train val test; do
  python scripts/extract_features.py \
      --encoder vjepa2 --fusion_mode late --pooling_strategy simple_attn \
      --weights_path  ./mr_rate_results/MrRate.5000.pt \
      --data_folder   /path/to/data \
      --jsonl_file    /path/to/reports.jsonl \
      --labels_file   $LAB/mrrate_merged_labels.csv \
      --splits_csv    $LAB/splits.csv \
      --split $SPLIT --normalizer zscore \
      --out_dir ./lp_features_majority
done

# 2) train the linear head (nn.Linear(dim, 14)) and report test AUROC
python scripts/linear_probe.py \
    --features_dir ./lp_features_majority \
    --results_dir  ./lp_results_majority
```

Or use `submit_linear_probe.sh` in this folder (set `WEIGHTS_PATH`,
`DATA_FOLDER`, `JSONL_FILE`).

Reproduce the labels: `python ../build_merged_group_labels.py --source majority`.
