#!/usr/bin/env bash
# Score one MR DINO checkpoint on THIS node with the validation MIL probe:
#   extract (one worker per GPU) -> patient-level K folds in parallel -> out-of-fold summary.
# Usage: DATA_FOLDER=<extracted coreg tree> probe_node.sh CHECKPOINT OUT_DIR [extra extract args]
#   CHECKPOINT: DCP step directory or the teacher .pt written by `python -m mr_dino.probe export`.
# Tokens go to node-local /tmp (FEATURE_DIR) and are deleted afterwards unless KEEP_FEATURES=1.
# Inside a running training allocation: srun --overlap --jobid JOB -N1 -n1 -w NODE probe_node.sh ...
set -euo pipefail

CKPT=${1:?checkpoint}; OUT=${2:?output dir}; shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE=${CODE:-"$(dirname "$SCRIPT_DIR")"}
DINOV3_ROOT=${DINOV3_ROOT:-/hnvme/workspace/b180dc51-sezgin/dinov3}
SIF=${SIF:-/hnvme/workspace/b180dc51-sezgin/mrrate.sif}
EXTRA_PIP=${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}
: "${DATA_FOLDER:?Set DATA_FOLDER to the extracted MR-RATE tree (batchXX/<study>/coreg_img)}"
GPUS=${GPUS_PER_NODE:-4}
FOLDS=${FOLDS:-5}
FEAT=${FEATURE_DIR:-/tmp/mrdino_probe_${SLURM_JOB_ID:-$$}_$(basename "$CKPT" .pt)}
LABEL_ARGS=()
[[ -n "${LABELS_CSV:-}" ]] && LABEL_ARGS+=(--labels-csv "$LABELS_CSV")
[[ -n "${SPLITS_CSV:-}" ]] && LABEL_ARGS+=(--splits-csv "$SPLITS_CSV")
export PYTHONPATH="${EXTRA_PIP}:${DINOV3_ROOT}:${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

probe() {
  singularity exec --nv -B /hnvme/workspace:/hnvme/workspace,/tmp:/tmp "$SIF" \
    python3 -m mr_dino.probe "$@"
}
wait_all() {   # fail if any background worker failed
  local status=0
  for pid in "$@"; do wait "$pid" || status=1; done
  return $status
}

mkdir -p "$OUT" "$FEAT"
echo "[probe_node] $(date '+%F %T') checkpoint=$CKPT features=$FEAT out=$OUT host=$(hostname)"
pids=()
for g in $(seq 0 $((GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$g RANK=$g WORLD_SIZE=$GPUS probe extract --checkpoint "$CKPT" \
    --data-folder "$DATA_FOLDER" --features-dir "$FEAT" "${LABEL_ARGS[@]}" "$@" \
    > "$OUT/extract_rank$g.log" 2>&1 &
  pids+=($!)
done
wait_all "${pids[@]}" || { echo "[probe_node] extraction failed; see $OUT/extract_rank*.log" >&2; exit 1; }
echo "[probe_node] $(date '+%F %T') extraction done ($(du -sh "$FEAT" | cut -f1))"

pids=()
for f in $(seq 0 $((FOLDS - 1))); do
  CUDA_VISIBLE_DEVICES=$((f % GPUS)) probe cv --features-dir "$FEAT" --out-dir "$OUT" \
    --fold "$f" --folds "$FOLDS" "${LABEL_ARGS[@]}" > "$OUT/cv_fold$f.log" 2>&1 &
  pids+=($!)
done
wait_all "${pids[@]}" || { echo "[probe_node] CV failed; see $OUT/cv_fold*.log" >&2; exit 1; }
probe summarize --features-dir "$FEAT" --out-dir "$OUT" --folds "$FOLDS" "${LABEL_ARGS[@]}" | tee "$OUT/summary.txt"
[[ "${KEEP_FEATURES:-0}" == 1 ]] || rm -rf "$FEAT"
