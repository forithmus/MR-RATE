#!/bin/bash
# Linear probe on the 14 merged-group majority labels in this folder.
#   extract_features.py (train/val/test) -> linear_probe.py
# Class count (14) is derived from the labels CSV automatically.
#SBATCH --job-name=mrrate_lp_merged
#SBATCH --output=logs/lp_merged_%j.out
#SBATCH --error=logs/lp_merged_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

# --- paths you must set ---
WEIGHTS_PATH=${WEIGHTS_PATH:-"./mr_rate_results/MrRate.5000.pt"}
DATA_FOLDER=${DATA_FOLDER:-"/path/to/data"}
JSONL_FILE=${JSONL_FILE:-"/path/to/reports.jsonl"}

# --- model config (must match the checkpoint used) ---
ENCODER=${ENCODER:-"vjepa2"}
FUSION_MODE=${FUSION_MODE:-"late"}
POOLING_STRATEGY=${POOLING_STRATEGY:-"simple_attn"}
NORMALIZER=${NORMALIZER:-"zscore"}
SPACE=${SPACE:-"native_space"}

# --- locate this folder, the repo scripts, and the label files ---
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$LAB_DIR/../.." && pwd)"          # contrastive-pretraining/scripts
FEAT_DIR=${FEAT_DIR:-"$SCRIPTS_DIR/lp_features_majority"}
RESULTS_DIR=${RESULTS_DIR:-"$SCRIPTS_DIR/lp_results_majority"}
mkdir -p "$LAB_DIR/logs"

cd "$SCRIPTS_DIR"

# 1) cache frozen-encoder features once per split
for SPLIT in train val test; do
  echo "=== extract_features: $SPLIT ==="
  python extract_features.py \
      --encoder "$ENCODER" --fusion_mode "$FUSION_MODE" \
      --pooling_strategy "$POOLING_STRATEGY" \
      --weights_path "$WEIGHTS_PATH" \
      --data_folder "$DATA_FOLDER" \
      --jsonl_file "$JSONL_FILE" \
      --labels_file "$LAB_DIR/mrrate_merged_labels.csv" \
      --splits_csv  "$LAB_DIR/splits.csv" \
      --split "$SPLIT" --space "$SPACE" --normalizer "$NORMALIZER" \
      --out_dir "$FEAT_DIR"
done

# 2) train the linear head and report test metrics
echo "=== linear_probe ==="
python linear_probe.py --features_dir "$FEAT_DIR" --results_dir "$RESULTS_DIR"
echo "Done. Results in $RESULTS_DIR"
