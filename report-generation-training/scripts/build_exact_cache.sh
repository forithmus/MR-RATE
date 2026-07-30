#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="/hnvme/workspace/b180dc51-sezgin/extra-pip:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m mrrate_report_training.build_cache \
  --config "${MRRATE_REPORT_CONFIG:-configs/base.yaml}" --split "${1:?split required}"
