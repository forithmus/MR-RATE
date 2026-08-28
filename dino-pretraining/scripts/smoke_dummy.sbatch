#!/usr/bin/env bash
#SBATCH --job-name=mr_dino_dummy
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=mrdino_%x_%j.out
#SBATCH --error=mrdino_%x_%j.err
set -euo pipefail

if [[ -z "${CODE:-}" ]]; then
  if [[ -d "$SLURM_SUBMIT_DIR/mr_dino" ]]; then
    CODE="$SLURM_SUBMIT_DIR"
  elif [[ -d "$SLURM_SUBMIT_DIR/dino-pretraining/mr_dino" ]]; then
    CODE="$SLURM_SUBMIT_DIR/dino-pretraining"
  else
    echo "Cannot locate dino-pretraining; set CODE explicitly" >&2
    exit 2
  fi
fi
SCRIPT_DIR="$CODE/scripts"
DUMMY=/tmp/mrdino_dummy_${SLURM_JOB_ID}
OUTPUT=${OUTPUT:-"$PWD/mrdino_dummy_${SLURM_JOB_ID}"}
SIF=${SIF:-/hnvme/workspace/b180dc51-sezgin/mrrate.sif}
EXTRA_PIP=${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}
export PYTHONPATH="${EXTRA_PIP}:${PYTHONPATH:-}"
singularity exec -B /hnvme/workspace:/hnvme/workspace,/tmp:/tmp \
  "$SIF" \
  python3 "$CODE/tests/make_dummy_cache.py" --out "$DUMMY" --studies 8 --shape 8 32 32

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29672}
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 "$SCRIPT_DIR/train_node.sh" \
  --preprocessed-dir "$DUMMY" --space atlas_space --output-dir "$OUTPUT" \
  --stage pretrain --arch tiny --steps 2 --batch-size 1 --grad-accum-steps 1 \
  --workers 0 --max-studies 2 --local-crops 2 \
  --global-shape 8 32 32 --local-shape 4 16 16 \
  --dino-prototypes 64 --ibot-prototypes 64 \
  --dino-head-hidden-dim 128 --ibot-head-hidden-dim 128 \
  --dino-bottleneck-dim 64 --ibot-bottleneck-dim 64 \
  --warmup-steps 1 --teacher-warmup-steps 1 \
  --save-every 1 --keep-checkpoints 2 --log-every 1 --resume none \
  --activation-checkpointing --no-selective-checkpointing --no-compile --no-fp8

test -f "$OUTPUT/checkpoints/step_00000002/COMPLETE"
echo "Synthetic MR DINO FSDP2 integration passed: $OUTPUT"
