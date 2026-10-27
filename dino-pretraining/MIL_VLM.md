# MR DINO → MIL and VLM

Multi-sequence MRI classification and streamed image–text contrastive training.
The image path trains the final DINO block and a visual projector; the text path
uses CXR-BERT. Two-pass pooling preserves gradients across the full study bag.

Training is intended to run at NVIDIA. No full MRI training was started on the
development cluster.

## Inputs and geometry

- Each study includes every eligible MRI sequence, encoded separately. All
  sequence tokens enter one study bag; sequence logits are never averaged.
- Auxiliary brain/defacing/segmentation masks are excluded from MRI inputs.
- Preprocessing uses the existing MR-RATE loader: canonical RAS, resampling,
  z-score normalization, posterior shift, and crop/padding. Encoder inputs are
  unchanged. Pooling support is derived before normalization, so normalized air
  and zero crop padding are not mistaken for anatomy.
- The checkpoint determines its model source, architecture, prefix tokens,
  physical RoPE, spacing, and training tile size. There is no CT block-24 or
  1024-dimensional assumption. Non-overlapping tiles cover the complete volume.
- The two H+ releases explicitly record `atlas_space`. The legacy 7B highres
  release states atlas space in its README but omits full run arguments. Its
  preprocessing fallback is the documented default at its pinned MR-RATE commit
  `43c05c61bf98d00c4e825a27a0fca3c742754d78`. Confirm original run overrides;
  an optional `preprocessing.json` beside its weights overrides those defaults.

Production MIL uses the final **74-class MRI diagnosis labels**, not the
14-group development labels. Set `LABELS` to the final 74-class CSV, with
`study_uid` plus the 74 diagnosis columns. The bundled ordered schema is
`data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/07_mri_diagnosis_extraction/data/mri_diagnoses_74.json`.
The loader rejects missing or extra classes and reorders CSV columns to this
canonical order. The checkpoint records that class order. Values below zero or
NaN are unknown and masked; 0/1 are supervised.

Train/validation/test patients must be disjoint. Test data is not used in training
or normalization. MRI anatomy routes may be supplied explicitly; CT organ
definitions are not imported. The default route allows all supported MRI tokens.

## Environment

Use Python 3.10+, the official DINOv3 checkout pinned by the release, and a
compatible CUDA PyTorch environment:

```bash
python -m pip install -r dino-pretraining/requirements.txt
export PYTHONPATH="$PWD/dino-pretraining:$DINOV3_ROOT:$PYTHONPATH"
git lfs pull --include=contrastive-pretraining/data/findings_sentences.jsonl
```

`DINOV3_ROOT` must point to official DINOv3 commit
`6876159a11b4df116f30f667f8c9888617df0751`. Training fails clearly on a sentence
file that is still a Git LFS pointer. Stage large models once per node to a local
cache with a completion marker, rather than copying them per GPU.

The examples below use eight ranks on one node. For multiple nodes, use the
site's `torchrun --nnodes ... --node-rank ... --master-addr ... --master-port ...`
settings. All ranks must see the same complete image/cache cohort. First perform
a two-node smoke in NVIDIA's actual runtime before a large launch; development
validation does not establish NVIDIA multi-node throughput or NCCL behavior.

## Offline prefix cache

Set `RELEASE`, `DATA`, `SPLITS`, `LABELS`, and `CACHE` to NVIDIA paths. Prefix
tokens, including each tile's prefix/context tokens, are stored as FP16. The
trainable last block is replayed during training. No token or sequence cap is
applied. This cache can be reused by MIL and VLM from the same DINO release.
It is large; use online mode if storage is constrained.

```bash
torchrun --standalone --nproc-per-node=8 -m mr_dino.mil cache \
  --release "$RELEASE" --data-folder "$DATA" --splits-csv "$SPLITS" \
  --split train --cache-dir "$CACHE"
torchrun --standalone --nproc-per-node=8 -m mr_dino.mil cache \
  --release "$RELEASE" --data-folder "$DATA" --splits-csv "$SPLITS" \
  --split val --cache-dir "$CACHE"
```

Edit all paths in `configs/vlm_mri.json`. Compute the shared normalization
using **training studies only**, with the DINO-initialized final block:

```bash
torchrun --standalone --nproc-per-node=8 -m mr_dino.vlm normalize \
  --config dino-pretraining/configs/vlm_mri.json \
  --output /path/to/mri_normalization.pt
```

For a fully online workflow set `mode` to `online`; normalization then reads raw
images instead of a prefix cache.

## MIL

The trainable final MRI block feeds normalized features and global-grid Fourier
positions to a gated classify-then-aggregate scalar head. The class softmax
spans all tokens of all sequences. Two passes give exact gradients without
retaining all transformer graphs. Localization is positive
`attention × token logit × output scale` evidence, excluding the output bias.

```bash
torchrun --standalone --nproc-per-node=8 -m mr_dino.mil train \
  --mode offline --release "$RELEASE" --cache-dir "$CACHE" \
  --normalization /path/to/mri_normalization.pt \
  --labels-csv "$LABELS" --splits-csv "$SPLITS" \
  --head-lr 0.002 --block-lr 0.00001 --epochs 10 --out runs/mri_mil
```

For online MIL replace `--mode offline --cache-dir "$CACHE"` with
`--mode online --data-folder "$DATA"`. The prefix remains frozen; the final
block and MIL head train. Online and offline use the same support and geometry.
Incomplete labelled train/validation cohorts fail by default; the explicit
`--allow-incomplete-data` option is for implementation smoke tests only.

MIL writes validation probabilities, per-class/macro AUROC and complete
optimizer/model checkpoints at epoch boundaries. `--resume epoch_001.pt`
continues from the next epoch; it does not claim mid-epoch replay. Learning rates
are explicit constant settings in this CLI. The example values are starting
settings, not a claim of optimized MRI performance.

## VLM

```bash
torchrun --standalone --nproc-per-node=8 -m mr_dino.vlm train \
  --config dino-pretraining/configs/vlm_mri.json --output runs/mri_vlm
```

Both `mode: offline` and `mode: online` implement the same objective:

1. DINO-initialized trainable final block, fixed train-only mean/RMS,
   global-grid Fourier features, and an MRI sequence-kind embedding.
2. CXR-BERT CLS plus projection into the shared space.
3. Exact streamed VL-CABS pooling over all sequence tokens, with learned pooling
   and contrastive temperatures.
4. the symmetric multi-positive InfoNCE: a study's sentences are all positives;
   its own other positive sentences do not become image-to-text negatives.

The supplied learning-rate groups, cosine/warmup schedule, 34-sentence sampling,
and 128-token sentence encoding are explicit in the example configuration. These limits are
on contrastive finding statements, not generated report lengths. All visual
tokens are retained. The VLM starts its last block from DINO, independently of
MIL. MRI sequence kinds replace CT window-kind embeddings. Unknown sequence
names use an explicit `other` kind.

Training covers eligible train studies without repeating padding examples;
inactive tail ranks are masked from positives and negatives. A singleton tail
contains no contrastive negative and has zero contrastive loss. Validation is
run each epoch on the validation cohort. Checkpoints include the optimizer,
next batch, temperatures, and per-rank RNG. `latest.pt` is resumed automatically
only within the explicitly supplied output directory. Recipe/world changes fail
instead of silently changing the resumed experiment. `SIGUSR1` requests a
checkpoint and continues training; `SIGTERM` requests a checkpoint and exits.
The trainer contains no Slurm cancellation, allocation-release, or 24-hour kill
timer.

## Development validation and previews

- Dense and two-pass MIL outputs/gradients agree, including gradients through
  trainable features and soft routes.
- The MRI adapter matches the backbone's final block with dynamic prefix/grid.
- MR-RATE streamed VL-CABS matches dense outputs and every gradient, including both
  temperatures and text/visual paths.
- A two-rank Gloo step matches the dense global contrastive objective.
- A synthetic full CLI test exercised train-only normalization, training,
  validation, inactive tail ranks, and interruption/resume. Resumed final block,
  projector, BERT and temperatures were bitwise identical to the uninterrupted
  control.
- A real five-sequence MRI study exercised final-block/MIL backward and a weight
  update. This was an implementation check, not a clinical training result.

Run unit checks with:

```bash
python -m pytest -q dino-pretraining/tests/test_mil_stream.py \
  dino-pretraining/tests/test_vlm_stream.py
```

`python -m mr_dino.preview --release ... --manifest ... --out ...` produces
whole-volume three-component PCA, fitted independently per sequence after tiles
are assembled, with physical voxel aspect ratios. The example manifest contains
`study_uid` and every `nifti_files` path. Prototype heads are absent from the
downloaded inference releases; the preview does not invent prototype maps.

Full synthetic two-rank normalization/train/validation/resume smoke (CPU, no model download):

```bash
bash dino-pretraining/scripts/smoke_mil_vlm.sh /tmp/mrrate_vlm_smoke
```

Use a fresh smoke directory. It compares resumed weights bitwise with an uninterrupted control.
