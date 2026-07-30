#!/usr/bin/env bash
#SBATCH --job-name=mr_syn_e2e
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/synthetic_gpu_e2e_%j.out
#SBATCH --error=logs/synthetic_gpu_e2e_%j.err
set -euo pipefail

project=/hnvme/workspace/b180dc51-sezgin/MR-RATE-report-training
sif=/hnvme/workspace/b180dc51-sezgin/mrrate-ib3.sif
mkdir -p "$project/logs"
cache="/tmp/mrrate_synthetic_${SLURM_JOB_ID}"
mkdir -p "$cache"/{cuda,triton,inductor,xdg}
export CUDA_CACHE_PATH="$cache/cuda"
export TRITON_CACHE_DIR="$cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export XDG_CACHE_HOME="$cache/xdg"
export MRRATE_REPORT_STREAM_THRESHOLD=128
export MRRATE_REPORT_STREAM_CHUNK=64

exec singularity exec --nv -B /hnvme:/hnvme,/tmp:/tmp "$sif" bash -lc "
  set -euo pipefail
  export PYTHONPATH=/hnvme/workspace/b180dc51-sezgin/extra-pip:$project/src
  export CUDA_CACHE_PATH=$CUDA_CACHE_PATH
  export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
  export TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR
  export XDG_CACHE_HOME=$XDG_CACHE_HOME
  export MRRATE_REPORT_STREAM_THRESHOLD=$MRRATE_REPORT_STREAM_THRESHOLD
  export MRRATE_REPORT_STREAM_CHUNK=$MRRATE_REPORT_STREAM_CHUNK
  cd $project
  python tests/gpu_e2e.py
"
