#!/usr/bin/env bash
# One stage of the MR writer pipeline on DFW (nodes x 8 H100), as a serial array of 4 h tasks that resume.
#
#   STAGE=predict  sbatch scripts/dfw/run_stage.sbatch      # MIL logits of train/val/test (resumable shards)
#   STAGE=calibrate sbatch --nodes=1 --array=1 scripts/dfw/run_stage.sbatch
#   STAGE=sft      sbatch scripts/dfw/run_stage.sbatch      # resumes from $OUT/sft/latest.pt
#   STAGE=tag      sbatch --nodes=1 scripts/dfw/run_stage.sbatch   # vLLM sentence tags (needs VLLM_PYTHON)
#   STAGE=pairs    sbatch --nodes=1 --array=1 scripts/dfw/run_stage.sbatch
#   STAGE=dpo      sbatch scripts/dfw/run_stage.sbatch      # from $OUT/sft/checkpoint_epoch001.pt
#   STAGE=generate SPLIT=val CHECKPOINT=$OUT/dpo/step_000200.pt sbatch --array=1 scripts/dfw/run_stage.sbatch
#
# Paths come from configs/paths.env (copy configs/paths.env.example). Every task checks for the stage's completion
# marker first and cancels the rest of its array when the stage is done.
#
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --job-name=mr_writer
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=1-19%1
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
HERE=${HERE:-$SLURM_SUBMIT_DIR}
# shellcheck disable=SC1091
source "$HERE/configs/paths.env"
STAGE=${STAGE:?set STAGE}
ARRAY_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
mkdir -p "$OUT"
done_marker() {
  case "$STAGE" in
    predict) [[ -f "$OUT/pred/COMPLETE" ]] ;;
    calibrate) [[ -f "$OUT/calibration.json" ]] ;;
    sft) [[ -f "$OUT/sft/COMPLETE.json" ]] ;;
    tag) [[ -f "$OUT/tags/COMPLETE" ]] ;;
    pairs) [[ -f "$OUT/pairs.sqlite" ]] ;;
    dpo) [[ -f "$OUT/dpo/COMPLETE.json" ]] ;;
    *) false ;;
  esac
}
if done_marker; then echo "$STAGE already complete"; scancel --state=PENDING "$ARRAY_ID" 2>/dev/null; exit 0; fi

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(getent ahostsv4 "$head_node" | awk '{print $1; exit}')
export MASTER_PORT=${MASTER_PORT:-$((29000 + SLURM_JOB_ID % 1000))}
export PYTHONPATH="$HERE/src:$BUNDLE/code/MR-RATE_8a55c42/dino-pretraining:$BUNDLE:$FORA:$FORA/report-generation:$DINOV3_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCHRUN=(srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1
          bash -c 'exec python3 -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc-per-node=8 \
                   --node-rank="$SLURM_NODEID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" -m "$@"' _)
COMMON=(--bundle "$BUNDLE" --data_folder "$DATA" --splits_csv "$SPLITS" --labels_csv "$LABELS" --reports "$REPORTS"
        --calibration "$OUT/calibration.json" --dense "$OUT/dense_logits.npz" --llm_path "$LLM"
        --save_every_minutes 20)
echo "=== $STAGE task ${SLURM_ARRAY_TASK_ID:-1} job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST $(date)"
case "$STAGE" in
  predict)
    # every rank finishes its whole study list unless the task ends at the time limit; a clean exit completes the stage
    "${TORCHRUN[@]}" mrrate_report.predict --bundle "$BUNDLE" --data-folder "$DATA" --splits-csv "$SPLITS" --out "$OUT/pred" \
      && touch "$OUT/pred/COMPLETE" ;;
  calibrate)
    python3 -m mrrate_report.calibrate --pred "$OUT/pred" --labels-csv "$LABELS" --out "$OUT/calibration.json" \
      --dense "$OUT/dense_logits.npz" ;;
  sft)
    RESUME=(); [[ -f "$OUT/sft/latest.pt" ]] && RESUME=(--resume "$OUT/sft/latest.pt")
    "${TORCHRUN[@]}" mrrate_report.train_sft "${COMMON[@]}" --results_dir "$OUT/sft" --epochs "${EPOCHS:-1}" "${RESUME[@]}" ;;
  tag)
    : "${VLLM_PYTHON:?set VLLM_PYTHON to a python with vLLM}"
    srun --ntasks=8 --gpus-per-task=1 "$VLLM_PYTHON" -m mrrate_report.tag_sentences --reports "$REPORTS" \
      --splits-csv "$SPLITS" --calibration "$OUT/calibration.json" --model "$LLM" --out "$OUT/tags" && touch "$OUT/tags/COMPLETE" ;;
  pairs)
    python3 -m mrrate_report.dpo_pairs --tags "$OUT/tags" --reports "$REPORTS" --calibration "$OUT/calibration.json" \
      --dense "$OUT/dense_logits.npz" --splits-csv "$SPLITS" --labels-csv "$LABELS" --out "$OUT/pairs.sqlite" ;;
  dpo)
    RESUME=(); [[ -f "$OUT/dpo/latest.pt" ]] && RESUME=(--resume "$OUT/dpo/latest.pt")
    "${TORCHRUN[@]}" mrrate_report.train_dpo "${COMMON[@]}" --results_dir "$OUT/dpo" --sft "${SFT:-$OUT/sft/checkpoint_epoch001.pt}" \
      --dpo_pairs "$OUT/pairs.sqlite" --max_steps "${DPO_STEPS:-600}" "${RESUME[@]}" ;;
  generate)
    "${TORCHRUN[@]}" mrrate_report.generate --checkpoint "${CHECKPOINT:?set CHECKPOINT}" --llm_path "$LLM" \
      --split "${SPLIT:-val}" --out "$OUT/generated_${SPLIT:-val}_$(basename "${CHECKPOINT%.pt}").csv" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
rc=$?
echo "=== $STAGE exited $rc $(date)"
if done_marker || [[ "$STAGE" == generate && $rc -eq 0 ]]; then scancel --state=PENDING "$ARRAY_ID" 2>/dev/null; fi
exit $rc
