#!/usr/bin/env bash
#SBATCH --job-name=mr_full_stack
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/gpu_full_stack_%j.out
#SBATCH --error=logs/gpu_full_stack_%j.err
set -euo pipefail

# Slurm executes a spooled copy of this script; BASH_SOURCE does not point
# at the repo, so the project path must be absolute.
project=/hnvme/workspace/b180dc51-sezgin/mrrate-new/report-generation-training
sif=/hnvme/workspace/b180dc51-sezgin/mrrate-ib3.sif
vllm_sif=/hnvme/workspace/b180dc51-sezgin/vllm2.sif
qwen=/hnvme/data/LLM-TZ/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
workspace="${WORKSPACE:-$project/runs/gpu_full_stack_${SLURM_JOB_ID}}"
mkdir -p "$project/logs" "$workspace"

cache="/tmp/mrrate_full_stack_${SLURM_JOB_ID}"
mkdir -p "$cache"/{cuda,triton,inductor,xdg}
export CUDA_CACHE_PATH="$cache/cuda"
export TRITON_CACHE_DIR="$cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export XDG_CACHE_HOME="$cache/xdg"

echo "=== stage 1: real Gemma writer + checkpoint round trip + generate CLI ==="
if [ "${SKIP_STAGE1:-0}" = 1 ]; then
  echo "stage 1 skipped (SKIP_STAGE1=1, reusing $workspace)"
else
singularity exec --nv -B /hnvme:/hnvme,/tmp:/tmp "$sif" bash -lc "
  set -euo pipefail
  export PYTHONPATH=/hnvme/workspace/b180dc51-sezgin/extra-pip:$project/src
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export CUDA_CACHE_PATH=$CUDA_CACHE_PATH TRITON_CACHE_DIR=$TRITON_CACHE_DIR
  export TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME=$XDG_CACHE_HOME
  export MRRATE_FULLSTACK_DIR=$workspace
  cd $project
  python tests/gpu_full_stack_e2e.py
"
fi

echo "=== stage 2 (best-effort): real vLLM pathology extraction with Qwen3.5-9B ==="
vllm_status=FAIL
set +e
singularity exec --nv -B /hnvme:/hnvme,/tmp:/tmp "$vllm_sif" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$project/src
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export XDG_CACHE_HOME=$XDG_CACHE_HOME
  cd $project
  python3 -m mrrate_report_training.extract_labels \
    --backend vllm \
    --generated-csv $workspace/generated_test.csv \
    --diagnoses-json $workspace/diagnoses_dummy74.json \
    --output-csv $workspace/pred_labels_test_vllm.csv \
    --work-dir $workspace/vllm_extraction_pred \
    --model-name $qwen
  python3 -m mrrate_report_training.extract_labels \
    --backend vllm \
    --generated-csv $workspace/generated_test.csv \
    --text-column findings_gt \
    --diagnoses-json $workspace/diagnoses_dummy74.json \
    --output-csv $workspace/control_labels_test_vllm.csv \
    --work-dir $workspace/vllm_extraction_control \
    --model-name $qwen
" && vllm_status=OK
set -e
echo "vllm extraction: $vllm_status"
echo "$vllm_status" > "$workspace/vllm_status.txt"

if [ "$vllm_status" = OK ]; then
  echo "=== stage 3: evaluation against vLLM-extracted labels ==="
  singularity exec -B /hnvme:/hnvme,/tmp:/tmp "$sif" bash -lc "
    set -euo pipefail
    export PYTHONPATH=/hnvme/workspace/b180dc51-sezgin/extra-pip:$project/src
    cd $project
    python -m mrrate_report_training.evaluate_reports \
      --generated-csv $workspace/generated_test.csv \
      --gt-labels $workspace/gt_labels.csv \
      --pred-labels $workspace/pred_labels_test_vllm.csv \
      --output-dir $workspace/eval_test_vllm
    python -m mrrate_report_training.clinical_metrics \
      --gt-labels $workspace/gt_labels.csv \
      --pred-labels $workspace/control_labels_test_vllm.csv \
      --output-dir $workspace/eval_control_vllm
  "
fi

echo "=== done: $workspace ==="
