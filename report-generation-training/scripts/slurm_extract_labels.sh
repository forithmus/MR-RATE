#!/usr/bin/env bash
#SBATCH --job-name=mrrate_extract_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# Required env: GENERATED_CSV, OUTPUT_CSV, WORK_DIR
# Optional env: PROJECT_DIR, DIAGNOSES_JSON, MODEL_NAME, CLASSIFIER_DIR,
#               EXTRA_PIP, SIF
set -euo pipefail
: "${GENERATED_CSV:?}"
: "${OUTPUT_CSV:?}"
: "${WORK_DIR:?}"

if [[ -n "${PROJECT_DIR:-}" ]]; then
  project_dir="$PROJECT_DIR"
elif [[ -d "${SLURM_SUBMIT_DIR:-}/src/mrrate_report_training" ]]; then
  project_dir="$SLURM_SUBMIT_DIR"
elif [[ -d "${SLURM_SUBMIT_DIR:-}/report-generation-training/src/mrrate_report_training" ]]; then
  project_dir="$SLURM_SUBMIT_DIR/report-generation-training"
else
  project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -d "$project_dir/src/mrrate_report_training" ]]; then
  echo "Cannot locate report-generation-training; set PROJECT_DIR explicitly" >&2
  exit 2
fi
diagnoses_json="${DIAGNOSES_JSON:-$project_dir/../data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/07_neurovfm_diagnosis_extraction/data/neurovfm_mri_diagnoses.json}"
extra_pip="${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}"
host_pythonpath="$extra_pip:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$host_pythonpath"
cd "$project_dir"

runtime=(python3)
if [[ -n "${SIF:-}" ]]; then
  runtime=(
    apptainer exec --nv -B /hnvme:/hnvme
    --env "PYTHONPATH=$project_dir/src"
    "$SIF" python3
  )
elif ! python3 -c 'import vllm' >/dev/null 2>&1; then
  echo "vLLM is unavailable; set SIF to a vLLM Apptainer image" >&2
  exit 2
fi

read -r -a generated_csvs <<< "$GENERATED_CSV"
command=(
  "${runtime[@]}" -m mrrate_report_training.extract_labels
  --backend vllm
  --generated-csv "${generated_csvs[@]}"
  --diagnoses-json "$diagnoses_json"
  --output-csv "$OUTPUT_CSV"
  --work-dir "$WORK_DIR"
)
if [[ -n "${MODEL_NAME:-}" ]]; then
  command+=(--model-name "$MODEL_NAME")
fi
if [[ -n "${CLASSIFIER_DIR:-}" ]]; then
  command+=(--classifier-dir "$CLASSIFIER_DIR")
fi
"${command[@]}"
