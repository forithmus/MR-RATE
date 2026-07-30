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

## Intentional training policy

- One source-grounded raw findings target per study, preserving line order.
- Raw findings cover all 97,896 split IDs. The encoder/MIL-compatible cohort is
  the same 97,887 studies retained by upstream MR-RATE's JSONL loader.
- Natural one-pass coverage: every train study occurs exactly once per epoch.
- No replacement sampler and no pathology oversampling.
- No MIL proposal dropout.
- No localization target, localization token, or localization loss.
- No MR-specific disease loss.
- The 1,536-token target ceiling preserves every normal findings report. It
  bounds one corrupted 32k-token outlier (`5NIUCVXWHA`) containing embedded
  editing dialogue.
- The encoder and MIL head are frozen.
- Exact cached training rejects `max_tokens_per_study != 0`.
- Startup requires the encoder SHA-256 and configuration recorded by MIL
  training to match. Cached MIL additionally requires the training-cache
  fingerprint to match the report token cache.

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

## 7. Integration tests

```bash
python tests/smoke_checks.py
sbatch scripts/synthetic_gpu_e2e.sbatch
```

The tests cover natural findings targets, strict cache validation,
online/cached numerical equivalence, frozen MIL behavior, single-prefix
construction, optimizer updates, strict weight provenance, and checkpoint
resume. The Slurm test uses a deterministic dummy dataset and does not replace
the required real-data smoke.
