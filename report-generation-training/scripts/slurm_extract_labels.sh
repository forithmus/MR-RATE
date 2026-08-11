#!/usr/bin/env bash
#SBATCH --job-name=mrrate_extract_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# Required env: GENERATED_CSV, PATHOLOGIES_JSON, OUTPUT_CSV, WORK_DIR
# Optional env: MODEL_NAME, CLASSIFIER_DIR, EXTRA_PIP
set -euo pipefail
: "${GENERATED_CSV:?}"
: "${PATHOLOGIES_JSON:?}"
: "${OUTPUT_CSV:?}"
: "${WORK_DIR:?}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
extra_pip="${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}"
export PYTHONPATH="$extra_pip:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

python -m mrrate_report_training.extract_labels \
  --backend vllm \
  --generated-csv $GENERATED_CSV \
  --pathologies-json "$PATHOLOGIES_JSON" \
  --output-csv "$OUTPUT_CSV" \
  --work-dir "$WORK_DIR" \
  ${MODEL_NAME:+--model-name "$MODEL_NAME"} \
  ${CLASSIFIER_DIR:+--classifier-dir "$CLASSIFIER_DIR"}
