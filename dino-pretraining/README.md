# Atlas-space 3-D DINOv3 for MR-RATE

This module adapts FORA's CT DINOv3 training strategy to MR-RATE's
**atlas-registered multi-sequence MRI**. It is self-supervised: reports and
pathology labels are not used.

## Data and view contract

Production training consumes the cache already produced by
`contrastive-pretraining/scripts/preprocess_volumes.py`:

```text
<preprocessed_dir>/atlas_space/_manifest.json
<preprocessed_dir>/atlas_space/<study_uid>.npz
    volumes: [N_sequences, D, H, W]
```

The loader rejects native/coreg-space caches, a missing manifest, mismatched
voxel spacing, malformed arrays, unexpected shapes, empty studies, and
non-finite values. Raw atlas-registered NIfTIs are also supported by the DDP debug
entry point using the same RAS, physical resampling, z-score normalization,
posterior shift, and crop/pad policy as MR-RATE preprocessing. Cached input is
recommended for production so NIfTI decoding cannot starve the GPUs.

MR-RATE-atlas already contains atlas-registered NIfTIs. The NPZ step below does
**not** register them again; it is the same optional offline packing used by the
previous MR-RATE training code to avoid repeatedly decoding and resampling very
large NIfTIs. Production launchers use the packed atlas cache for throughput.
Unlike coregistration alone, atlas registration also places corresponding
anatomy from different studies in the same reference coordinate system.

Every sequence in every retained study is an anchor exactly once per local
data epoch. Cache files are assigned to distributed ranks by sequence count
(bytes break ties), not merely by file count. A sequence receives weight
`1 / sequences_in_study`, globally rescaled to mean one, so studies with many
sequences do not dominate the objective. Studies are shuffled, and their
sequence anchors stay grouped in a shuffled within-study order; each worker
keeps only its current study stack in memory, avoiding repeated NPZ decoding.

For each anchor:

1. Global view 1 uses the anchor sequence.
2. Global view 2 uses a different atlas-aligned sequence with probability 0.75
   (or the anchor when no second sequence exists). The two crops overlap by at
   least 75% on every axis.
3. Local crops stay inside the global intersection and may come from any
   aligned sequence.
4. DINO and KoLeo therefore learn anatomy shared across MR contrasts.
5. Each iBOT teacher/student target always uses the **identical sequence,
   spatial crop, and patch order**. Cross-sequence images are never used as
   patch-level targets.

There are no flips or axis reversals. Student-only MR augmentations are bounded
gain/bias, noise, and mild 3-D blur. The clean teacher receives no distortion.

The backbone and losses match the CT implementation: 3-D convolutional patch
embed, physical 3-D RoPE, DINO global/local self-distillation, iBOT block
masking, distributed KoLeo, optional Gram anchoring, EMA teacher, atomic full
state checkpoints, exact sampler/RNG resume, FSDP2, BF16, optional H200 FP8,
and compile caches on node-local `/tmp`.

## Prepare the atlas-space cache

From `contrastive-pretraining/`:

```bash
python scripts/preprocess_volumes.py \
  --data_folder /path/to/MR-RATE-atlas/mri \
  --out_dir /path/to/mrrate_preprocessed \
  --space atlas_space \
  --normalizer zscore \
  --num_workers 8
```

The default cache geometry is `(1.0, 0.5, 0.5) mm` and
`256 x 384 x 384`. MR DINO uses a `(2, 16, 16)` patch kernel, normal global
crops of `64 x 192 x 192`, and local crops of `32 x 96 x 96`. The high-resolution
stage uses `64 x 384 x 384` globals.

## Dependencies

The volumetric model uses the official DINOv3 source. The tested checkout is:

```text
https://github.com/facebookresearch/dinov3
commit 6876159a11b4df116f30f667f8c9888617df0751
```

Set `DINOV3_ROOT` to that checkout and add both it and this directory to
`PYTHONPATH`.

## Synthetic tests

The unit/integration suite creates a dummy atlas-registered multi-sequence cache and
checks manifest rejection, deterministic cross-sequence alignment, complete
sequence indexing, local/global containment, 3-D masks, a real DINO+iBOT
forward/backward optimizer step, EMA update, and checkpoint reload:

```bash
cd dino-pretraining
PYTHONPATH=/path/to/dinov3:$PYTHONPATH pytest -q
```

Run the actual four-GPU FSDP2 path on a generated dummy dataset:

```bash
sbatch scripts/smoke_dummy.sbatch
```

The job succeeds only after `step_00000002/COMPLETE` exists.

## Production training

The production launcher is deliberately gated on a prebuilt atlas-space cache:

```bash
cd dino-pretraining
PREPROCESSED_DIR=/path/to/mrrate_preprocessed \
SPLITS_CSV=/path/to/splits.csv \
OUTPUT=/path/to/mrdino3d_7b/pretrain \
sbatch scripts/train_32n_7b.sbatch
```

Important overrides are `STEPS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `WORKERS`,
`LOCAL_CROPS`, `CROSS_SEQUENCE_PROBABILITY`, `WARMUP_STEPS`, `RESUME`, `FP8`,
and `COMPILE`. The default warmup is 1,000 optimizer steps. `RESUME=latest`
loads the latest checkpoint only when its `COMPLETE` marker exists.

For later stages:

```bash
STAGE=gram RESUME=/path/to/pretrain/checkpoints/step_00100000 \
PREPROCESSED_DIR=/path/to/mrrate_preprocessed \
OUTPUT=/path/to/mrdino3d_7b/gram \
sbatch scripts/train_32n_7b.sbatch
```

The launcher catches the Slurm wall-time signal, writes a complete distributed
checkpoint with per-rank sampler and RNG state, and requeues the same job. Do
not start production before `scripts/smoke_dummy.sbatch` passes in the same
container and DINOv3 checkout.

The same 32-node H+ fallback used by CT-DINO is also available:

```bash
PREPROCESSED_DIR=/path/to/mrrate_preprocessed \
OUTPUT=/path/to/mrdino3d_hplus/pretrain \
sbatch scripts/train_32n_hplus.sbatch
```

Both production launchers include the established bad-node exclusions, Slurm
mail notifications, 24-hour requeue policy, deadline checkpointing, node-local
compiler caches, InfiniBand/NCCL settings, and optional split filtering. The
7B path is the production target; H+ is a smaller DDP fallback.

## Lightweight/debug run

`mr_dino.train_ddp` supports one or more GPUs and a tiny backbone:

```bash
torchrun --standalone --nproc-per-node=1 -m mr_dino.train_ddp \
  --preprocessed-dir /path/to/mrrate_preprocessed \
  --output-dir /tmp/mrdino_debug \
  --arch tiny --steps 10 --workers 0 --local-crops 2 \
  --prototypes 64 --head-hidden-dim 128 \
  --warmup-steps 1 --teacher-warmup-steps 1 \
  --save-every 5 --log-every 1 --resume none
```

This path is for correctness/debugging. Use FSDP2 for the 7B production model.
