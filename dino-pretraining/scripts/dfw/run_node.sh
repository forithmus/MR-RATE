#!/usr/bin/env bash
# One-node torchrun for mr_dino MIL/VLM commands on DFW (8 H100), in the miniconda `mrrate` env.
#   scripts/run_node.sh <python module> <args...>      e.g. run_node.sh dfw_mil.mil_fast ...
set -euo pipefail

TRAIN=/lustre/fsw/portfolios/healthcareeng/users/dongy/vlm/mr-rate/training
WORKDIR=${WORKDIR:-${TRAIN}/mil_hplus_atlas_highres}
CODE=${CODE:-${WORKDIR}/code/MR-RATE/dino-pretraining}
DINOV3_ROOT=${DINOV3_ROOT:-${TRAIN}/dinov3_6876159}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

eval "$(/lustre/fsw/portfolios/healthcareeng/users/dongy/miniconda3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-mrrate}"

# load_release exec()s the release's mr_dino/model.py; without this Python would write
# __pycache__ into the (read-only to us) release folder.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
# mr_dino and dfw_mil both come from CODE (this checkout).
export PYTHONPATH="${CODE}:${DINOV3_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

module=$1; shift
exec python3 -m torch.distributed.run --standalone --nproc-per-node="${GPUS_PER_NODE}" -m "$module" "$@"
