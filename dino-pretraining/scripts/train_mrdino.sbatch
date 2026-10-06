#!/usr/bin/env bash
#SBATCH --job-name=mr_dino3_hplus
#SBATCH --partition=h200
#SBATCH --nodes=4
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
OUTPUT=${OUTPUT:-"$PWD/mrdino3d_${ARCH:-hplus}/$STAGE"}
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/REQUEUE_REQUESTED"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29675}
end_ts=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*EndTime=\([0-9T:-]*\).*/\1/p' | head -1)
if [[ -n "$end_ts" ]]; then
  export MRDINO_DEADLINE_EPOCH=$(( $(date -d "$end_ts" +%s) - 1200 ))
fi

# Phases 2/3 restart the schedule at step 0, so they need their own settings (FORA CT-DINO
# phase2_gram_recipe.args / phase3_hires_recipe.args): short warmup, teacher temperature fixed at
# 0.07, EMA 0.996, no prototype freeze. Phase 3 has 4x the tokens per global crop -> batch 4 + checkpointing.
# --lr is the peak at global batch 1024 and is sqrt-scaled to the real batch.
# H+ memory/speed per H200 (1-node benchmark 943261, real crop sizes, compiled, bf16):
#   phase-1 crops: batch 8 no checkpointing OOM; batch 8 + selective checkpointing 0.39 steps/s 32 GiB;
#   batch 4 no checkpointing 0.95 steps/s 83 GiB.
#   highres crops (Gram on): batch 2 no checkpointing 128 GiB (too tight); batch 2 + checkpointing
#   0.23 steps/s 43 GiB; batch 4 + checkpointing 0.10 steps/s 81 GiB.
# Defaults keep the FORA global batch per phase on 4 nodes: 128 (phases 1-2), 64 (phase 3).
ARCH=${ARCH:-hplus}
FIXED_TEACHER=(--teacher-temperature 0.07 --teacher-warmup-temperature 0.07 --teacher-warmup-steps 1
               --teacher-momentum 0.996 --freeze-last-layer-steps 0)
case "$STAGE" in
  pretrain)   # ~10 sequence-epochs at 4 nodes (batch 128); the cosine decay ends here
    STEPS=${STEPS:-50000}; GRAM_WEIGHT=0; BATCH_SIZE=${BATCH_SIZE:-8}; LOCAL_CROPS=${LOCAL_CROPS:-8}
    ACT_CKPT=${ACT_CKPT:-1}
    STAGE_ARGS=() ;;
  gram)       # Gram anchoring to the incoming teacher
    STEPS=${STEPS:-10000}; GRAM_WEIGHT=${GRAM_WEIGHT:-1.0}; BATCH_SIZE=${BATCH_SIZE:-8}; LOCAL_CROPS=${LOCAL_CROPS:-8}
    ACT_CKPT=${ACT_CKPT:-1}
    STAGE_ARGS=(--lr 5e-4 --warmup-steps 250 --weight-decay 0.1 --weight-decay-end 0.4 "${FIXED_TEACHER[@]}") ;;
  highres)    # 64x384x384 globals (whole axial plane, 18,432 tokens), 4 locals 32x192x192
    STEPS=${STEPS:-5000}; GRAM_WEIGHT=${GRAM_WEIGHT:-1.5}; BATCH_SIZE=${BATCH_SIZE:-4}; LOCAL_CROPS=${LOCAL_CROPS:-4}
    ACT_CKPT=${ACT_CKPT:-1}
    STAGE_ARGS=(--lr 4e-4 --warmup-steps 500 --weight-decay 0.04 --weight-decay-end 0.2 "${FIXED_TEACHER[@]}") ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
if [[ "$STAGE" != pretrain && "${RESUME:-latest}" == latest && ! -f "$OUTPUT/checkpoints/latest.txt" ]]; then
  echo "STAGE=$STAGE starts from the previous stage: set RESUME=<previous stage>/checkpoints/step_N" >&2
  exit 2
fi
[[ "${COMPILE:-1}" == 1 ]] && COMPILE_ARGS=(--compile) || COMPILE_ARGS=(--no-compile)
[[ "$ACT_CKPT" == 1 ]] && CKPT_ARGS=(--activation-checkpointing) || CKPT_ARGS=(--no-activation-checkpointing)
[[ "${FP8:-1}" == 1 ]] && FP8_ARGS=(--fp8) || FP8_ARGS=(--no-fp8)
SPLIT_ARGS=()
if [[ -n "${SPLITS_CSV:-}" ]]; then
  SPLIT_ARGS=(--splits-csv "$SPLITS_CSV")
fi

echo "MR DINOv3 $ARCH (FORA recipe): stage=$STAGE nodes=$SLURM_NNODES output=$OUTPUT data=$DATA_FOLDER"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 "$SCRIPT_DIR/train_node.sh" \
  --data-folder "$DATA_FOLDER" \
  --space "${SPACE:-coreg_space}" \
  "${SPLIT_ARGS[@]}" \
  --output-dir "$OUTPUT" --stage "$STAGE" --arch "$ARCH" --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --workers "${WORKERS:-8}" --local-crops "$LOCAL_CROPS" \
  --cross-sequence-probability "${CROSS_SEQUENCE_PROBABILITY:-0.25}" \
  --cross-view-sequences "${CROSS_VIEW_SEQUENCES:-same}" \
  --resume "${RESUME:-latest}" --gram-weight "$GRAM_WEIGHT" \
  "${COMPILE_ARGS[@]}" --no-fp8 "${CKPT_ARGS[@]}" "${STAGE_ARGS[@]}"
rc=$?
if [[ -f "$OUTPUT/REQUEUE_REQUESTED" ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$rc"
