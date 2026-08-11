#!/usr/bin/env bash
# Usage: generate_reports.sh <mode> <split> <config> <checkpoint> [extra args]
# Example: GPUS unused; run on one allocated GPU node.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="/hnvme/workspace/b180dc51-sezgin/extra-pip:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
mode="${1:?mode required (online|cached)}"
split="${2:?split required (val|test)}"
config="${3:?config required}"
checkpoint="${4:?checkpoint required}"
exec python -m mrrate_report_training.generate \
  --mode "$mode" --split "$split" --config "$config" \
  --checkpoint "$checkpoint" \
  --output-csv "runs/generated_${split}.csv" "${@:5}"
