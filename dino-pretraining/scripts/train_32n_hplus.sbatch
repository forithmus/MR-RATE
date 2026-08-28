#!/usr/bin/env bash
#SBATCH --job-name=mr_dino3_hplus
#SBATCH --partition=h200
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --exclude=h21-24,h23-24,h24-17,h23-18
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@1200
#SBATCH --output=mrdino_%x_%j.out
#SBATCH --error=mrdino_%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sezgin.er@dqbm.uzh.ch
set -uo pipefail

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
: "${PREPROCESSED_DIR:?Set PREPROCESSED_DIR to the MR-RATE preprocessing cache root}"
STAGE=${STAGE:-pretrain}
OUTPUT=${OUTPUT:-"$PWD/mrdino3d_hplus/$STAGE"}
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/REQUEUE_REQUESTED"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29673}
end_ts=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*EndTime=\([0-9T:-]*\).*/\1/p' | head -1)
if [[ -n "$end_ts" ]]; then
  export MRDINO_DEADLINE_EPOCH=$(( $(date -d "$end_ts" +%s) - 1200 ))
fi

case "$STAGE" in
  pretrain) STEPS=${STEPS:-100000}; GRAM_WEIGHT=0 ;;
  gram) STEPS=${STEPS:-10000}; GRAM_WEIGHT=${GRAM_WEIGHT:-1.0} ;;
  highres) STEPS=${STEPS:-5000}; GRAM_WEIGHT=${GRAM_WEIGHT:-1.0} ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
SPLIT_ARGS=()
if [[ -n "${SPLITS_CSV:-}" ]]; then
  SPLIT_ARGS=(--splits-csv "$SPLITS_CSV")
fi

echo "MR DINOv3 H+: stage=$STAGE nodes=$SLURM_NNODES output=$OUTPUT"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 env TRAIN_MODULE=mr_dino.train_ddp "$SCRIPT_DIR/train_node.sh" \
  --preprocessed-dir "$PREPROCESSED_DIR" --space atlas_space "${SPLIT_ARGS[@]}" \
  --output-dir "$OUTPUT" --stage "$STAGE" --arch hplus --steps "$STEPS" \
  --batch-size "${BATCH_SIZE:-1}" --workers "${WORKERS:-2}" \
  --local-crops "${LOCAL_CROPS:-8}" \
  --cross-sequence-probability "${CROSS_SEQUENCE_PROBABILITY:-0.75}" \
  --prototypes 65536 --head-hidden-dim 4096 \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --teacher-warmup-steps "${TEACHER_WARMUP_STEPS:-1000}" \
  --save-every 250 --keep-checkpoints 3 --resume "${RESUME:-latest}" \
  --gram-weight "$GRAM_WEIGHT" --activation-checkpointing
rc=$?
if [[ -f "$OUTPUT/REQUEUE_REQUESTED" ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$rc"
