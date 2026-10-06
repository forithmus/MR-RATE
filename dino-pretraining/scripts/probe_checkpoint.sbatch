#!/usr/bin/env bash
#SBATCH --job-name=mr_dino_probe
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=mrdino_%x_%j.out
#SBATCH --error=mrdino_%x_%j.err
# Usage: DATA_FOLDER=<coreg tree> sbatch probe_checkpoint.sbatch CHECKPOINT OUT_DIR [extra extract args]
set -euo pipefail
SCRIPT_DIR=${SCRIPT_DIR:-"$SLURM_SUBMIT_DIR/scripts"}
[[ -x "$SCRIPT_DIR/probe_node.sh" ]] || SCRIPT_DIR="$SLURM_SUBMIT_DIR/dino-pretraining/scripts"
exec "$SCRIPT_DIR/probe_node.sh" "$@"
