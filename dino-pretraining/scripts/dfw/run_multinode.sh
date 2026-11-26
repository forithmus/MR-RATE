#!/usr/bin/env bash
# Per-node torchrun for multi-node MIL on DFW (8 H100 per node); launched once per node by srun.
#   srun ... scripts/run_multinode.sh <python module> <args...>
set -euo pipefail

TRAIN=/lustre/fsw/portfolios/healthcareeng/users/dongy/vlm/mr-rate/training
WORKDIR=${WORKDIR:-${TRAIN}/mil_hplus_atlas_highres}
CODE=${CODE:-${WORKDIR}/code/MR-RATE/dino-pretraining}
DINOV3_ROOT=${DINOV3_ROOT:-${TRAIN}/dinov3_6876159}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

eval "$(/lustre/fsw/portfolios/healthcareeng/users/dongy/miniconda3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-mrrate}"

# load_release exec()s the release's mr_dino/model.py; keep Python from writing __pycache__ there.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
# mr_dino and dfw_mil both come from CODE (this checkout).
export PYTHONPATH="${CODE}:${DINOV3_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-10}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

module=$1; shift
exec python3 -m torch.distributed.run \
  --nnodes="${SLURM_NNODES}" --nproc-per-node="${GPUS_PER_NODE}" --node-rank="${SLURM_NODEID}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  -m "$module" "$@"
