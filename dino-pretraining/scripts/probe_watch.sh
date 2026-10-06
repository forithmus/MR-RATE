#!/usr/bin/env bash
# Probe chosen steps of a running MR DINO stage as soon as their checkpoints appear.
# Usage (login node, e.g. under nohup): DATA_FOLDER=<coreg tree> probe_watch.sh RUN_DIR STEP [STEP ...]
#   RUN_DIR is a stage output (…/mrdino3d_hplus/pretrain). For each step the EMA-teacher backbone is
#   exported at once to RUN_DIR/probe/step_N/teacher.pt (checkpoint rotation cannot delete it), then
#   probe_checkpoint.sbatch is submitted. Results: RUN_DIR/probe/step_N/results.json + summary.txt.
set -euo pipefail
RUN=${1:?run dir}; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE=${CODE:-"$(dirname "$SCRIPT_DIR")"}
SIF=${SIF:-/hnvme/workspace/b180dc51-sezgin/mrrate.sif}
EXTRA_PIP=${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}
DINOV3_ROOT=${DINOV3_ROOT:-/hnvme/workspace/b180dc51-sezgin/dinov3}
: "${DATA_FOLDER:?Set DATA_FOLDER}"
export DATA_FOLDER PYTHONPATH="${EXTRA_PIP}:${DINOV3_ROOT}:${CODE}:${PYTHONPATH:-}"
pending=("$@")
while ((${#pending[@]})); do
  left=()
  for step in "${pending[@]}"; do
    ckpt=$(printf '%s/checkpoints/step_%08d' "$RUN" "$step")
    out="$RUN/probe/step_$step"
    if [[ -f "$ckpt/COMPLETE" ]]; then
      mkdir -p "$out"
      singularity exec -B /hnvme/workspace:/hnvme/workspace "$SIF" \
        python3 -m mr_dino.probe export --checkpoint "$ckpt" --out "$out/teacher.pt"
      job=$(cd "$CODE" && sbatch --parsable --chdir="$CODE" --output="$out/slurm_%j.out" \
        --error="$out/slurm_%j.err" "$SCRIPT_DIR/probe_checkpoint.sbatch" "$out/teacher.pt" "$out")
      echo "[probe_watch] $(date '+%F %T') step $step exported, probe job $job"
    else
      left+=("$step")
    fi
  done
  pending=("${left[@]+"${left[@]}"}")
  ((${#pending[@]})) && sleep 300
done
echo "[probe_watch] all steps submitted"
