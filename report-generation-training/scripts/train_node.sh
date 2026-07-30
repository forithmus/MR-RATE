#!/usr/bin/env bash
set -euo pipefail
: "${MODE:?MODE must be online or cached}"
: "${CONFIG:?CONFIG is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
extra_pip="${EXTRA_PIP:-/hnvme/workspace/b180dc51-sezgin/extra-pip}"
export PYTHONPATH="$extra_pip:$project_dir/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cache_root="/tmp/mrrate_report_${SLURM_JOB_ID:-local}"
mkdir -p "$cache_root"/{cuda,triton,inductor,xdg,model}
export CUDA_CACHE_PATH="$cache_root/cuda"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export XDG_CACHE_HOME="$cache_root/xdg"

stage_path() {
  local source="$1" name="$2" destination="$cache_root/model/$name"
  (
    flock 9
    if [[ -d "$source" ]]; then
      if [[ ! -f "$destination/.complete" ]]; then
        mkdir -p "$destination"
        cp -aL "$source"/. "$destination"/
        touch "$destination/.complete"
      fi
    else
      mkdir -p "$destination"
      local target="$destination/$(basename "$source")"
      if [[ ! -f "$target" || $(stat -Lc %s "$target") -ne $(stat -Lc %s "$source") ]]; then
        cp -aL "$source" "$target"
      fi
      printf '%s\n' "$target"
    fi
  ) 9>"$cache_root/model/$name.lock"
}

llm_local="$cache_root/model/llm"
stage_path "$LLM_PATH" llm >/dev/null
encoder_local="$(stage_path "$ENCODER_CHECKPOINT" encoder)"
mil_local="$(stage_path "$MIL_CHECKPOINT" mil)"

nnodes="${SLURM_NNODES:-1}"
node_rank="${SLURM_NODEID:-0}"
gpus="${GPUS_PER_NODE:-4}"
command=(python -m torch.distributed.run \
  --nnodes="$nnodes" \
  --node-rank="$node_rank" \
  --nproc-per-node="$gpus" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m mrrate_report_training.train \
  --mode "$MODE" \
  --config "$CONFIG" \
  --llm-path "$llm_local" \
  --encoder-checkpoint "$encoder_local" \
  --mil-checkpoint "$mil_local")
if [[ -n "${RESUME:-}" ]]; then
  command+=(--resume "$RESUME")
fi
if [[ "${MAX_STUDIES:-0}" -gt 0 ]]; then
  command+=(--max-studies "$MAX_STUDIES")
fi
if [[ "${MAX_UPDATES:-0}" -gt 0 ]]; then
  command+=(--max-updates "$MAX_UPDATES")
fi
exec "${command[@]}"
