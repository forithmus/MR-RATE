#!/usr/bin/env bash
#SBATCH --job-name=mrrate_report
#SBATCH --partition=h200
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/mrrate_report_%j.out
#SBATCH --error=logs/mrrate_report_%j.err
set -euo pipefail

: "${MODE:?set MODE=online or cached}"
: "${CONFIG:?set CONFIG to an absolute yaml path}"
: "${LLM_PATH:?set LLM_PATH}"
: "${ENCODER_CHECKPOINT:?set ENCODER_CHECKPOINT}"
: "${MIL_CHECKPOINT:?set MIL_CHECKPOINT}"
case "$MODE" in online|cached) ;; *) exit 2 ;; esac

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sif="${SIF:-/hnvme/workspace/b180dc51-sezgin/mrrate-ib3.sif}"
mkdir -p "$project_dir/logs"
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT="${MASTER_PORT:-29541}"
export MODE CONFIG LLM_PATH ENCODER_CHECKPOINT MIL_CHECKPOINT
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MAX_STUDIES="${MAX_STUDIES:-0}"
export MAX_UPDATES="${MAX_UPDATES:-0}"
export RESUME="${RESUME:-}"

exec srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  singularity exec --nv -B /hnvme:/hnvme,/tmp:/tmp "$sif" \
  bash "$project_dir/scripts/train_node.sh"
