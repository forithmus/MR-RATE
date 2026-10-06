#!/usr/bin/env bash
#SBATCH --job-name=mr_dino3_vitl
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
: "${DATA_FOLDER:?Set DATA_FOLDER to the extracted MR-RATE-coreg NIfTI root (batchXX/<study>/coreg_img)}"
STAGE=${STAGE:-pretrain}
OUTPUT=${OUTPUT:-"$PWD/mrdino3d_vitl/$STAGE"}
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/REQUEUE_REQUESTED"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29675}
end_ts=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*EndTime=\([0-9T:-]*\).*/\1/p' | head -1)
if [[ -n "$end_ts" ]]; then
  export MRDINO_DEADLINE_EPOCH=$(( $(date -d "$end_ts" +%s) - 1200 ))
fi

case "$STAGE" in
  pretrain) STEPS=${STEPS:-30000}; GRAM_WEIGHT=0 ;;   # ~6 sequence-epochs at 4 nodes (batch 128); cosine ends here
  gram) STEPS=${STEPS:-10000}; GRAM_WEIGHT=${GRAM_WEIGHT:-1.0} ;;
  highres) STEPS=${STEPS:-5000}; GRAM_WEIGHT=${GRAM_WEIGHT:-1.5} ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
[[ "${COMPILE:-1}" == 1 ]] && COMPILE_ARGS=(--compile) || COMPILE_ARGS=(--no-compile)
[[ "${FP8:-1}" == 1 ]] && FP8_ARGS=(--fp8) || FP8_ARGS=(--no-fp8)
SPLIT_ARGS=()
if [[ -n "${SPLITS_CSV:-}" ]]; then
  SPLIT_ARGS=(--splits-csv "$SPLITS_CSV")
fi

echo "MR DINOv3 ViT-L (FORA recipe): stage=$STAGE nodes=$SLURM_NNODES output=$OUTPUT data=$DATA_FOLDER"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 "$SCRIPT_DIR/train_node.sh" \
  --data-folder "$DATA_FOLDER" \
  --space "${SPACE:-coreg_space}" \
  "${SPLIT_ARGS[@]}" \
  --output-dir "$OUTPUT" --stage "$STAGE" --arch large --steps "$STEPS" \
  --batch-size "${BATCH_SIZE:-8}" --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --workers "${WORKERS:-8}" --local-crops "${LOCAL_CROPS:-8}" \
  --cross-sequence-probability "${CROSS_SEQUENCE_PROBABILITY:-0.25}" \
  --cross-view-sequences "${CROSS_VIEW_SEQUENCES:-same}" \
  --resume "${RESUME:-latest}" --gram-weight "$GRAM_WEIGHT" \
  "${COMPILE_ARGS[@]}" --no-fp8 --no-activation-checkpointing
rc=$?
if [[ -f "$OUTPUT/REQUEUE_REQUESTED" ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$rc"
