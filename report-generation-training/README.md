# MR-RATE single-writer report training

This is a standalone report-generation training layer for MR-RATE. It does not
modify FORA or an existing MR-RATE checkout.

The writer target is the natural `findings` field from
`MR-RATE-validation/reports/all_reports.csv`. The standardized
`findings_sentences.jsonl` remains an encoder/MIL data dependency only; it is
never used as report-generation supervision. The trainer does not infer
abnormal/healthy labels or split reports into artificial subtasks.

The two execution modes are deliberately equivalent:

- `online`: a frozen MR-RATE encoder produces the complete projected visual
  token bag during training.
- `cached`: the same complete projected token bag is read from the upstream
  ragged memmap cache.

In both modes the same frozen 74-label `ClassifyThenAggregate` MIL head consumes
the full bag, a trainable query resampler makes 512 language-prefix tokens, and
one Gemma LoRA writer learns the complete findings text. The MIL probabilities
are soft conditioning context; they are not report targets.

## Required artifacts

The files have distinct roles:

| Input | Purpose |
| --- | --- |
| `all_reports.csv` | Writer supervision from `study_uid, findings` |
| `findings_sentences.jsonl` | Reproduce the encoder/MIL cohort only |
| labels CSV | Exact 74-class MIL schema |
| splits CSV | Train/validation/test membership |
| MR-RATE checkpoint | Frozen visual encoder and projection |
| `mil_head.pt` | Frozen Classify-Then-Aggregate MIL head and provenance |
| Gemma snapshot | Frozen language model plus trainable report LoRA |
| exact token cache | Required only for cached mode |

The real encoder checkpoint and `mil_head.pt` are intentionally placeholders
in `configs/base.yaml`; replace them before running.

## 1. Configure

```bash
cd /hnvme/workspace/b180dc51-sezgin/MR-RATE-report-training
cp configs/base.yaml configs/my_run.yaml
```

Edit at least:

```yaml
encoder_checkpoint: /absolute/path/to/the/mr-rate-checkpoint.pt
mil_checkpoint: /absolute/path/to/the/corresponding/mil_head.pt
llm_path: /absolute/path/to/gemma-3-12b-it

data:
  data_folder: /absolute/path/to/mri
  reports_csv: /absolute/path/to/all_reports.csv
  jsonl_file: /absolute/path/to/findings_sentences.jsonl
  labels_file: /absolute/path/to/mrrate_labels.csv
  splits_csv: /absolute/path/to/splits.csv
  cached_tokens_dir: /absolute/path/to/exact_tokens
```

Do not pair an arbitrary MIL head and encoder. Preflight checks the recorded
encoder SHA-256, architecture, label order, and cache fingerprint.

## 2. Run preflight

Inside the training container:

```bash
export PROJECT=/hnvme/workspace/b180dc51-sezgin/MR-RATE-report-training
export PYTHONPATH=/hnvme/workspace/b180dc51-sezgin/extra-pip:$PROJECT/src
cd "$PROJECT"

python -m mrrate_report_training.preflight \
  --config configs/my_run.yaml \
  --mode cached
```

Use `--mode online` for online training. Preflight rejects pooled features,
token-capped caches, mismatched dimensions/classes, missing findings, and
mismatched MIL/encoder provenance.

## 3. Prepare an exact cache, if needed

```bash
export MRRATE_REPORT_CONFIG="$PROJECT/configs/my_run.yaml"
bash scripts/build_exact_cache.sh train
bash scripts/build_exact_cache.sh val
```

Run cache generation on a CUDA node. It is an expensive frozen-encoder pass,
not part of every training epoch. The builder always uses
`max_tokens_per_study=0`. If the verified exact cache already exists, skip
this step.

## 4. Run a small real-data smoke

Use a separate output directory so the smoke cannot overwrite production
checkpoints:

```bash
cp configs/my_run.yaml configs/my_smoke.yaml
# Edit output_dir in configs/my_smoke.yaml, for example:
# output_dir: runs/mrrate_single_writer_smoke
```

From an allocated GPU node, inside the container:

```bash
GPUS_PER_NODE=1 bash scripts/train_cached.sh \
  configs/my_smoke.yaml \
  --max-studies 8 \
  --max-updates 2
```

For the online path:

```bash
GPUS_PER_NODE=1 bash scripts/train_online.sh \
  configs/my_smoke.yaml \
  --max-studies 2 \
  --max-updates 1
```

Online mode is slower because it loads/resamples MRI volumes and runs the
frozen encoder during every epoch. The online reader supports
`batchXX/<study>.zip`: it extracts only the current study under node-local
`/tmp` and removes it after encoding.

## 5. Submit Slurm training

The launcher defaults to two nodes with four GPUs per node. Command-line
`sbatch` options can override the node count.

```bash
export PROJECT=/hnvme/workspace/b180dc51-sezgin/MR-RATE-report-training
export MODE=cached
export CONFIG="$PROJECT/configs/my_run.yaml"
export LLM_PATH=/absolute/path/to/gemma-3-12b-it
export ENCODER_CHECKPOINT=/absolute/path/to/the/mr-rate-checkpoint.pt
export MIL_CHECKPOINT=/absolute/path/to/the/corresponding/mil_head.pt
export GPUS_PER_NODE=4
export MAX_STUDIES=0
export MAX_UPDATES=0

sbatch --nodes=2 "$PROJECT/scripts/slurm_train.sh"
```

For online training, set `MODE=online`. The node launcher stages Gemma and
both checkpoints once per node, creates node-local CUDA/Triton/Inductor
caches, then launches one `torchrun` rank per GPU.

The production settings in `configs/my_run.yaml` are:

```yaml
epochs: 1
batch_size: 1
gradient_accumulation: 1
learning_rate: 0.0001
```

Every real study is seen exactly once. No study is duplicated to fill a
distributed batch.

## 6. Resume

```bash
export RESUME=/absolute/path/to/checkpoint-00000500.pt
sbatch --nodes=2 "$PROJECT/scripts/slurm_train.sh"
```

Keep `MODE`, `CONFIG`, node count, and artifact variables the same. Checkpoints
contain the resampler/connector, report LoRA, optimizer, scheduler, next data
position, and per-rank RNG state.

## 7. Inference: generate reports for validation/test

Build the exact token cache for the target split first (or use online mode):

```bash
export MRRATE_REPORT_CONFIG="$PROJECT/configs/my_run.yaml"
bash scripts/build_exact_cache.sh val    # and/or test
```

Then, on an allocated GPU node:

```bash
bash scripts/generate_reports.sh cached val configs/my_run.yaml \
  runs/mrrate_single_writer_v1/last.pt
bash scripts/generate_reports.sh cached test configs/my_run.yaml \
  runs/mrrate_single_writer_v1/last.pt
```

This runs `python -m mrrate_report_training.generate`, which rebuilds the
exact training prefix (visual token bag -> query resampler, frozen MIL
conditioning tokens, report prompt), loads the trainable checkpoint state
with strict schema checks, and decodes greedily with the KV cache. Output is
`runs/generated_<split>.csv` with columns `study_uid, findings_gt,
findings_pred` plus a `.meta.json` provenance sidecar. Use `--mode online`
to encode volumes on the fly, `--max-new-tokens` to bound decoding, and
`--num-shards/--shard-index` to split a large set across independent GPU
jobs — each shard automatically writes its own
`...shardNNofMM.csv` (the evaluator accepts multiple CSVs). A preempted or
killed job can be requeued with `--resume` to append only missing studies;
an existing non-empty output is otherwise refused unless `--overwrite` is
given.

## 8. Evaluation: clinical accuracy and NLG metrics

Clinical accuracy re-extracts pathology labels from the generated reports
with the NeuroVFM MRI diagnosis prompt used to build the current 74-label
ground truth (`data-preprocessing/.../07_neurovfm_diagnosis_extraction`),
then compares them against the ground-truth labels CSV. The bundled schema
contains the exact 74 diagnosis keys and guidance used for the completed
MR-RATE extraction. On a GPU node with vLLM and Gemma 4 available:

```bash
export GENERATED_CSV="runs/generated_test.csv"
export OUTPUT_CSV="runs/pred_labels_test.csv"
export WORK_DIR="runs/label_extraction_test"
# When vLLM is provided through Apptainer:
# export SIF=/path/to/vllm.sif
sbatch scripts/slurm_extract_labels.sh
```

The launcher uses the bundled `neurovfm_mri_diagnoses.json` by default. Set
`DIAGNOSES_JSON` only to evaluate a deliberately different NeuroVFM-format
schema; the vLLM backend rejects the legacy 37-pathology JSON so the two
label definitions cannot be mixed silently.

Empty generated reports receive all-zero labels. The `keyword` backend of
`extract_labels` is a deterministic name/synonym matcher for tests only.

Finally, compute all metrics (no GPU needed):

```bash
python -m mrrate_report_training.evaluate_reports \
  --generated-csv runs/generated_test.csv \
  --gt-labels /path/to/mrrate_labels.csv \
  --pred-labels runs/pred_labels_test.csv \
  --output-dir runs/eval_test
```

Outputs: `metrics.json` (corpus BLEU-1..4, ROUGE-1/2/L F1, METEOR when nltk
wordnet data is present, plus the clinical summary), `per_pathology_metrics.csv`
(per-pathology TP/FP/TN/FN, sensitivity, specificity, precision, NPV, F1,
accuracy, balanced accuracy, prevalence), and `nlg_per_sample.csv`. The
clinical summary reports macro / micro / positive-support-weighted
aggregates, subset accuracy, and Hamming accuracy; undefined ratios (e.g.
specificity of a pathology with no negative studies) are null and excluded
from macro averages. Omit `--pred-labels` to score NLG only.

The CPU-runnable end-to-end trial of this whole chain on a fabricated
74-diagnosis dummy dataset:

```bash
python tests/e2e_dummy.py
```

## 9. Ablation: no classification-label conditioning

`writer.mil_conditioning` selects what the language model is conditioned on:

| | `all_classes` (default) | `none` (ablation) |
| --- | --- | --- |
| LLM prefix | `image_start` + visual query tokens + `image_end` + **74 MIL class tokens** + report prompt | `image_start` + visual query tokens + `image_end` + report prompt |
| Classification labels | One token per MIL class: label-name embedding + projected (probability, probability − threshold) from the frozen MIL head | **Not present — image visuals only** |
| MIL head / `mil_checkpoint` | Loaded frozen, verified against the encoder | Never loaded; `mil_checkpoint` may stay a placeholder |
| Report supervision | Natural `findings` text (identical in both modes) | Natural `findings` text (identical in both modes) |

To run the ablation experiment, make two configs that differ ONLY in
`mil_conditioning` and `output_dir`, train both, and evaluate both on the
same splits and metrics:

```bash
cp configs/my_run.yaml configs/my_run_ablation.yaml
# in configs/my_run_ablation.yaml set:
#   writer.mil_conditioning: none
#   output_dir: runs/mrrate_single_writer_ablation_v1

GPUS_PER_NODE=4 bash scripts/train_cached.sh configs/my_run.yaml
GPUS_PER_NODE=4 bash scripts/train_cached.sh configs/my_run_ablation.yaml

bash scripts/generate_reports.sh cached test configs/my_run.yaml \
  runs/mrrate_single_writer_v1/last.pt
bash scripts/generate_reports.sh cached test configs/my_run_ablation.yaml \
  runs/mrrate_single_writer_ablation_v1/last.pt
# then sections 9's extraction + evaluate_reports for each generated CSV;
# the metric deltas are the measured contribution of class conditioning.
```

Everything else is unchanged: both training modes (`online`/`cached`),
preflight, resume, sharded generation with `--resume`, and the evaluation
pipeline work identically, so ablation and full runs are directly comparable
on the same clinical accuracy and NLG metrics. Preflight/training/generation
skip the MIL artifacts and MIL provenance checks in `none` mode (encoder
loading verification still applies to online encoding).

Safety rails: an ablation writer refuses MIL inputs (and a full writer
refuses their absence); ablation checkpoints contain no
`label_embeddings`/`mil_*` tensors and carry a `mil_conditioning` stamp, so
loading an ablation checkpoint into a full-conditioning model — or a full
checkpoint (including pre-ablation ones, which default to `all_classes`)
into an ablation model — fails loudly instead of generating from
mismatched weights.

Tested by `tests/test_mil_ablation.py` (prefix composition, loss/decoding
without MIL, all mode-mixup errors, cross-mode checkpoint refusal in both
directions, config validation) and by `tests/gpu_ablation_probe.py`, a GPU
gate that trains and decodes the real Gemma writer in `none` mode through
the actual `generate.py` CLI and asserts a full-conditioning checkpoint is
refused. `tests/e2e_dummy.py` additionally runs an ablation writer through
checkpoint load, generation, label extraction, and the full per-pathology
clinical evaluation, asserting all classes are scored for ablation output
exactly as for full-conditioning output.
