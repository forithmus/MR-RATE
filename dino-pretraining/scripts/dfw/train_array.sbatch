#!/usr/bin/env bash
# Online MIL (74 classes) on DFW: resumable normalization, then resumable MIL training.
# Each 4 h interactive task checkpoints before its deadline; the next task continues. The last task
# of an array queues the next array. Writes only into $RUN.
#
#   sbatch scripts/dfw/train_array.sbatch                              # baseline into runs/mil_full
#   TRAIN_FRACTION=0.2 sbatch scripts/dfw/train_array.sbatch           # data-size ablation -> runs/mil_frac0.2
#   FUSION=cross_sequence NORMALIZATION=<baseline>/normalization.pt \
#     sbatch scripts/dfw/train_array.sbatch                            # cross-sequence attention -> runs/mil_cross_sequence
#   RELEASE=<modality-channel release> RUN=<dir> sbatch scripts/dfw/train_array.sbatch
#                                                                      # modality-channel backbone
#   SMOKE=1 sbatch --array=1 --time=01:30:00 scripts/dfw/train_array.sbatch   # 96+24 studies, 2 epochs
#
# CODE is the dino-pretraining folder of this checkout; mr_dino and dfw_mil are imported from it.
#
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --job-name=mil_hp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=1-19%1
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/users/dongy/vlm/mr-rate/training/mil_hplus_atlas_highres/logs/slurm/%x_%A_%a.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/users/dongy/vlm/mr-rate/training/mil_hplus_atlas_highres/logs/slurm/%x_%A_%a.err
set -uo pipefail

TRAIN=/lustre/fsw/portfolios/healthcareeng/users/dongy/vlm/mr-rate/training
WORKDIR=${WORKDIR:-${TRAIN}/mil_hplus_atlas_highres}
RELEASE=${RELEASE:-${DEFAULT_RELEASE}}
DATA=${DATA:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip/mri_atlas}
LABELS=${LABELS:-${TRAIN}/splits_neurovfm74/mrrate_neurovfm74_labels.csv}
CODE=${CODE:-${WORKDIR}/code/MR-RATE/dino-pretraining}
SPLITS=${SPLITS:-${CODE}/../contrastive-pretraining/scripts/eval_labels/splits_merged_majority/splits.csv}
TRAIN_FRACTION=${TRAIN_FRACTION:-1}
SUBSET_SEED=${SUBSET_SEED:-0}
FUSION=${FUSION:-none}
FUSION_LR=${FUSION_LR:-0.0001}
DEFAULT_RELEASE=${TRAIN}/share_bundles/mrdino3d_hplus_atlas_infer_highres_step5000
EPOCHS=${EPOCHS:-10}
HEAD_LR=${HEAD_LR:-0.002}
BLOCK_LR=${BLOCK_LR:-0.00001}
VAL_RESERVE=${VAL_RESERVE:-3600}
DEADLINE_MARGIN=${DEADLINE_MARGIN:-900}
CYCLE=${CYCLE:-1}
MAX_CYCLES=${MAX_CYCLES:-8}
SMOKE=${SMOKE:-0}
NAME=mil
[[ "$TRAIN_FRACTION" != 1 ]] && NAME+="_frac${TRAIN_FRACTION}"
[[ "$FUSION" != none ]] && NAME+="_${FUSION}"
[[ "$NAME" == mil ]] && NAME=mil_full
if [[ "$RELEASE" != "$DEFAULT_RELEASE" && -z "${RUN:-}" ]]; then
  echo "RELEASE differs from the baseline backbone: set RUN to a new run directory" >&2; exit 2
fi
if [[ "$SMOKE" == 1 ]]; then
  RUN=${RUN:-${WORKDIR}/runs/smoke_${NAME}_${SLURM_JOB_ID}}
  mkdir -p "$RUN"
  if [[ ! -f "$RUN/splits_smoke.csv" ]]; then
    PYTHONPATH="${CODE}" python3 "${CODE}/scripts/dfw/make_smoke_split.py" --splits-csv "$SPLITS" --labels-csv "$LABELS" \
      --data-folder "$DATA" --space atlas_space --out "$RUN/splits_smoke.csv" || exit 1
  fi
  SPLITS="$RUN/splits_smoke.csv"
  EPOCHS=2
fi
RUN=${RUN:-${WORKDIR}/runs/${NAME}}
NORMALIZATION=${NORMALIZATION:-${RUN}/normalization.pt}
export WORKDIR CODE
mkdir -p "$RUN"
TASK=${SLURM_ARRAY_TASK_ID:-1}
TASK_MAX=${SLURM_ARRAY_TASK_MAX:-1}
ARRAY_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
echo "=== ${SLURM_JOB_NAME} cycle ${CYCLE}/${MAX_CYCLES} task ${TASK}/${TASK_MAX} job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} $(date)"
echo "run=${RUN} release=${RELEASE} epochs=${EPOCHS} head_lr=${HEAD_LR} block_lr=${BLOCK_LR} smoke=${SMOKE}"
echo "code=${CODE} ($(git -C "${CODE}" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "train_fraction=${TRAIN_FRACTION} subset_seed=${SUBSET_SEED} fusion=${FUSION} fusion_lr=${FUSION_LR} normalization=${NORMALIZATION}"
echo "partition=${SLURM_JOB_PARTITION} account=${SLURM_JOB_ACCOUNT} world=$((SLURM_NNODES * 8)) (= studies per optimizer step)"

if [[ -f "$RUN/mil/TRAINING_COMPLETE" ]]; then
  echo "TRAINING_COMPLETE present; nothing to do."
  scancel --state=PENDING "$ARRAY_ID" 2>/dev/null || true
  exit 0
fi

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
export MASTER_PORT=${MASTER_PORT:-$((29000 + SLURM_JOB_ID % 1000))}
end_ts=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*EndTime=\([0-9T:-]*\).*/\1/p' | head -1)
export MIL_DEADLINE_EPOCH=$(( $(date -d "$end_ts" +%s) - DEADLINE_MARGIN ))
echo "checkpoint deadline $(date -d "@${MIL_DEADLINE_EPOCH}")"
export MIL_TIMING_LOG="$RUN/timing/task_${SLURM_JOB_ID}"
mkdir -p "$RUN/timing"
SRUN=(srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1
      "${CODE}/scripts/dfw/run_multinode.sh")

queue_next_cycle() {
  [[ "$SMOKE" == 1 ]] && return 0
  (( TASK == TASK_MAX )) || return 0
  if (( CYCLE >= MAX_CYCLES )); then
    echo "array exhausted at MAX_CYCLES=${MAX_CYCLES}; resubmit to continue."
    return 0
  fi
  local nid
  nid=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" \
    --account="${SLURM_JOB_ACCOUNT}" --partition="${SLURM_JOB_PARTITION}" --nodes="${SLURM_NNODES}" \
    --job-name="${SLURM_JOB_NAME}" \
    --export="ALL,CYCLE=$((CYCLE + 1)),MAX_CYCLES=${MAX_CYCLES},RUN=${RUN},EPOCHS=${EPOCHS},HEAD_LR=${HEAD_LR},BLOCK_LR=${BLOCK_LR},VAL_RESERVE=${VAL_RESERVE},CODE=${CODE},RELEASE=${RELEASE},NORMALIZATION=${NORMALIZATION},TRAIN_FRACTION=${TRAIN_FRACTION},SUBSET_SEED=${SUBSET_SEED},FUSION=${FUSION},FUSION_LR=${FUSION_LR}" \
    "${CODE}/scripts/dfw/train_array.sbatch") || { echo "WARNING: could not queue next cycle" >&2; return 0; }
  echo "queued cycle $((CYCLE + 1)) as array ${nid}"
  echo -e "$(date -Is)\tmil\tcycle $((CYCLE + 1))\tarray ${nid}\t(chained from ${SLURM_JOB_ID})" >> "${WORKDIR}/logs/submissions.log"
}

made_progress=0
if [[ ! -f "$NORMALIZATION" ]]; then
  echo "=== normalize (train studies only) $(date)"
  rm -f "$RUN/REQUEUE_REQUESTED"
  "${SRUN[@]}" dfw_mil.normalize_fast --release "$RELEASE" --data-folder "$DATA" --splits-csv "$SPLITS" \
    --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --output "$NORMALIZATION"
  if [[ -f "$RUN/REQUEUE_REQUESTED" ]]; then
    rm -f "$RUN/REQUEUE_REQUESTED" "$RUN/.failures"
    echo "normalization checkpointed at the deadline; the next task continues it."
    queue_next_cycle
    exit 0
  fi
  [[ -f "$NORMALIZATION" ]] || { echo "normalization failed" >&2; exit 1; }
  rm -f "$RUN/.failures"
  made_progress=1
fi

remaining=$(( MIL_DEADLINE_EPOCH - $(date +%s) ))
if (( remaining < 1800 )); then
  echo "only ${remaining}s left after normalization; MIL starts in the next task."
  queue_next_cycle
  exit 0
fi

echo "=== MIL train $(date)"
STOP_ARGS=()
[[ "$SMOKE" == 1 && ! -f "$RUN/mil/latest.pt" ]] && STOP_ARGS=(--stop-after-steps 3)
for attempt in 1 2; do
  rm -f "$RUN/mil/REQUEUE_REQUESTED"
  "${SRUN[@]}" dfw_mil.mil_fast --release "$RELEASE" --data-folder "$DATA" --normalization "$NORMALIZATION" \
    --labels-csv "$LABELS" --splits-csv "$SPLITS" --head-lr "$HEAD_LR" --block-lr "$BLOCK_LR" \
    --epochs "$EPOCHS" --val-reserve "$VAL_RESERVE" --out "$RUN/mil" \
    --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --fusion "$FUSION" --fusion-lr "$FUSION_LR" \
    "${STOP_ARGS[@]}"
  rc=$?
  if [[ -f "$RUN/mil/TRAINING_COMPLETE" ]]; then
    echo "MIL complete after ${EPOCHS} epochs."
    scancel --state=PENDING "$ARRAY_ID" 2>/dev/null || true
    exit 0
  fi
  if [[ -f "$RUN/mil/REQUEUE_REQUESTED" ]]; then
    rm -f "$RUN/mil/REQUEUE_REQUESTED" "$RUN/.failures"
    if [[ "$SMOKE" == 1 && $attempt == 1 ]]; then
      echo "smoke: stopped after 3 steps as intended; resuming in the same job to test resume."
      STOP_ARGS=()
      continue
    fi
    echo "MIL checkpointed at the deadline; the next task continues it."
    queue_next_cycle
    exit 0
  fi
  echo "MIL exited ${rc} without a checkpoint marker" >&2
  break
done
if (( made_progress == 0 )); then
  fails=$(( $(cat "$RUN/.failures" 2>/dev/null || echo 0) + 1 )); echo "$fails" > "$RUN/.failures"
  if (( fails >= 3 )); then
    echo "3 consecutive failed tasks; cancelling the rest of ${ARRAY_ID}." >&2
    scancel --state=PENDING "$ARRAY_ID" 2>/dev/null || true
  fi
fi
queue_next_cycle
exit 1
