#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE=${CODE:-"$(dirname "$SCRIPT_DIR")"}
DINOV3_ROOT=${DINOV3_ROOT:-/hnvme/workspace/b180dc51-sezgin/dinov3}
SIF=${SIF:-/hnvme/workspace/b180dc51-sezgin/mrrate.sif}
EXTRA_PIP=${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

mkdir -p /tmp/fora_cache/{nv,triton,inductor,xdg,mpl}
export CUDA_CACHE_PATH=/tmp/fora_cache/nv
export TRITON_CACHE_DIR=/tmp/fora_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/fora_cache/inductor
export XDG_CACHE_HOME=/tmp/fora_cache/xdg
export MPLCONFIGDIR=/tmp/fora_cache/mpl
export MRDINO_LOCAL_DATA_DIR=${MRDINO_LOCAL_DATA_DIR:-/tmp/mrdino3d_${SLURM_JOB_ID}/data}
mkdir -p "$MRDINO_LOCAL_DATA_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_MMAP_THRESHOLD_=131072
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-'=mlx5_2,mlx5_3,mlx5_4,mlx5_5'}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-10}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-'^ibp,ibs,ib,lo,docker'}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE:-20000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export PYTHONPATH="${EXTRA_PIP}:${DINOV3_ROOT}:${CODE}:${PYTHONPATH:-}"

ZIG_ARCHIVE=${ZIG_ARCHIVE:-/hnvme/workspace/b180dc51-sezgin/toolchains/zig-linux-x86_64-0.13.0.tar.xz}
ZIG_SHA256=d45312e61ebcc48032b77bc4cf7fd6915c11fa16e4aad116b66c9468211230ea
ZIG_PARENT=/tmp/fora_cache/zig
ZIG_ROOT="$ZIG_PARENT/zig-linux-x86_64-0.13.0"
if [[ ! -x "$ZIG_ROOT/zig" ]]; then
  printf '%s  %s\n' "$ZIG_SHA256" "$ZIG_ARCHIVE" | sha256sum --check --status || {
    echo "Missing or invalid pinned Zig archive: $ZIG_ARCHIVE" >&2
    exit 2
  }
  mkdir -p "$ZIG_PARENT"
  tar -xf "$ZIG_ARCHIVE" -C "$ZIG_PARENT"
fi
export MRDINO_ZIG_ROOT="$ZIG_ROOT"
export APPTAINERENV_MRDINO_ZIG_ROOT="$ZIG_ROOT"
export APPTAINERENV_CC="$SCRIPT_DIR/host_toolchain/zig-cc"
export SINGULARITYENV_MRDINO_ZIG_ROOT="$ZIG_ROOT"
export SINGULARITYENV_CC="$SCRIPT_DIR/host_toolchain/zig-cc"

singularity exec --nv \
  -B /hnvme/workspace:/hnvme/workspace,/tmp:/tmp \
  "$SIF" \
  python3 -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="$SLURM_NODEID" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    -m "${TRAIN_MODULE:-mr_dino.train_fsdp}" "$@"
