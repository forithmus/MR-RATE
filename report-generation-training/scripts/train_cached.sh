#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="/hnvme/workspace/b180dc51-sezgin/extra-pip:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m torch.distributed.run --standalone --nproc-per-node="${GPUS_PER_NODE:-1}" \
  -m mrrate_report_training.train --mode cached \
  --config "${1:-configs/base.yaml}" "${@:2}"
