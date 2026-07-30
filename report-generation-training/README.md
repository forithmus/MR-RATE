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

Set paths in `configs/base.yaml`:

1. MR-RATE pretraining checkpoint used by the trained MIL model.
2. The corresponding `mil_head.pt`.
3. A local Gemma 3 model.
4. `all_reports.csv`, plus the MR-RATE JSONL, labels CSV, and splits CSV used
   to reconstruct the frozen encoder/MIL cohort.
5. For cached mode, `token_features_{split}.json` plus its ragged memmap files.

Preflight rejects pooled features, token-capped caches, mismatched
dimensions/classes, missing reports, or mismatched MIL/encoder provenance:

```bash
python -m mrrate_report_training.preflight --config configs/base.yaml --mode cached
```

## Build exact caches

```bash
bash scripts/build_exact_cache.sh train
bash scripts/build_exact_cache.sh val
```

The cache builder always uses full projected token bags with
`max_tokens_per_study=0`.

## Train

```bash
bash scripts/train_online.sh configs/base.yaml
bash scripts/train_cached.sh configs/base.yaml
```

For a quick real-data test, add `--max-studies 2 --max-updates 1` to the Python
command in either launcher. Checkpoints contain the query
resampler/connector, report LoRA, optimizer/scheduler state, configuration,
data position, and per-rank RNG state.

For multi-node Slurm training, use `scripts/slurm_train.sh`. It launches one
`torchrun` agent per node, stages Gemma plus the encoder/MIL checkpoints once
per node, and places CUDA, Triton, and TorchInductor caches under node-local
`/tmp`.

The online reader supports the released `batchXX/<study>.zip` layout. It
extracts only the current study to node-local `/tmp`, applies upstream MR-RATE
preprocessing, encodes it, and removes the extraction.

## Tests

```bash
python tests/smoke_checks.py
```

The tests cover natural findings targets, strict cache validation,
online/cached numerical equivalence, frozen MIL behavior, single-prefix
construction, optimizer updates, strict weight provenance, and checkpoint
resume. `scripts/synthetic_gpu_e2e.sbatch` runs the integration gate on a
Slurm GPU with a deterministic dummy dataset.
