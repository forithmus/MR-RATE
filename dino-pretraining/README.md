# Co-registered 3-D DINOv3 for MR-RATE

This module adapts FORA's CT DINOv3 training strategy to MR-RATE's
**co-registered multi-sequence MRI** (`coreg_space`, default; `atlas_space` remains
selectable). It is self-supervised: reports and pathology labels are not used for
training; they are used only by the validation probe (below) to score checkpoints.

## Data and view contract

Production training reads an extracted MR-RATE-coreg NIfTI tree directly:

```text
<data_folder>/batchXX/<study_uid>/coreg_img/*.nii.gz     # --space coreg_space (default)
<data_folder>/batchXX/<study_uid>/atlas_img/*.nii.gz     # --space atlas_space
```

**Why coreg, not atlas.** Both spaces use the same rigid (ANTs, Mattes MI) registration, so
both are voxel-aligned across sequences, which the cross-sequence views need. They differ
in the target grid:
- **Coreg** keeps every sequence on the study's T1w center scan's native grid (e.g.
  0.47×0.47×1.2 mm, 190×240×240 mm field of view).
- **Atlas** resamples into the 1 mm MNI box (193×229×193 mm).

The model input is 1.0×0.5×0.5 mm, so from atlas the 0.5 mm in-plane detail is
interpolation, and the MNI box crops the skull base and neck (−12% head volume on a
checked study), which matters for the spinal labels. `native_space` is rejected
(its sequences are not aligned).

It imports the canonical discovery and per-volume preprocessing functions from
`contrastive-pretraining/scripts/data.py`, the loader used by previous MR-RATE
MIL training. Therefore atlas selection, canonical RAS orientation, physical
resampling, z-score normalization, posterior shift, and crop/padding are exactly
shared rather than reimplemented. `--space coreg_space` maps to `coreg_img`,
`--space atlas_space` to `atlas_img`. Checkpoints record their space, and resume refuses
to cross spaces (checkpoints written before the coreg switch are atlas-space).

An optional preprocessed volume cache remains available with
`--preprocessed-dir`. This is separate from cached MIL: cached MIL stores the
frozen encoder's output token bags, whereas the optional NPZ cache stores input
volumes before the encoder. MR-DINO does not require either cache.

Every sequence in every retained study is an anchor exactly once per local
data epoch. Studies are assigned to distributed ranks by sequence count
(source bytes break ties), not merely by study count. A sequence receives weight
`1 / sequences_in_study`, globally rescaled to mean one, so studies with many
sequences do not dominate the objective. Studies are shuffled, and their
sequence anchors stay grouped in a shuffled within-study order; each worker
keeps only its current study stack in memory, avoiding repeated NIfTI decoding.

For each anchor:

1. Global view 1 uses the anchor sequence.
2. Global view 2 uses a different atlas-aligned sequence with probability 0.25
   (`--cross-sequence-probability`; otherwise, or when no second sequence exists,
   the anchor). The two crops overlap by at least 25% on every axis.
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

## Recipe fixes ported from FORA CT-DINO (Sep 2026)

The CT pipeline this module was adapted from started with the same DINOv3-7B continuation
settings and failed; each item below is a diagnosed failure and the fix the working FORA
ViT-L run uses (code: `mr_dino/recipe.py`, `objective.py`, `data.py`; tests:
`tests/test_fora_recipe.py`). All are on by default.

| Problem seen in FORA | Fix (default) |
|---|---|
| DINOv3 plain-linear prototype layer (std 0.02) never sharpened at lr 5e-5 → flat Sinkhorn targets for 10k steps (iBOT target entropy 8.0/11.5 nats, top-1 0.006) | cosine prototypes on both heads (`--normalize-prototypes`) |
| 1k warmup + 7B settings drove positional drift in the last blocks | DINOv2 ViT-L recipe: lr 1e-3 at batch 1024 (sqrt-scaled), warmup 2,500, teacher-temperature warmup 0.04→0.07 over 10k, weight decay 0.04→0.4, EMA 0.992→1, AdamW β2 0.999, clip 3, layer-wise lr decay 0.9, patch-embed lr ×0.2, prototype layers frozen for 1,250 steps |
| 7B continuation was worse than a from-scratch ViT-L at the same budget | train from scratch at a trainable size: FORA used `--arch large` (ViT-L); MR uses `--arch hplus` (ViT-H+, 0.84B). 16k/16k prototypes, heads 2048/256, no FP8 |
| global crops overlapping ≥75% were near-duplicates | ≥25% overlap per axis (`--global-overlap`), locals inside one of the two globals |
| head-facing block encoded where a token sat in its crop (last-block position leakage 0.20, crop-quadrant prototype fields) | second global snapped to the patch lattice + **cross-crop twin-token objective** (`--cross-view-weight 1.0`): the student's patch prediction in one crop matches the teacher target of the same voxels in the other crop; leakage fell to 0.09 and stayed flat |
| patch targets balanced over all masked tokens at once | Sinkhorn within 2×2×2 spatial bins of the crop (`--position-bins`, CAPI) |
| raw loss hid whether the student learned (CE = H + KL) | `metrics.jsonl` logs `dino/ibot_target_entropy`, `_top1`, `_prototype_usage`, `dino_kl`, `ibot_kl`, `cross`, `cross_kl`, `cross_pairs` |

Already fixed here before the port: the distributed Sinkhorn uses the actual global token count.

**MR-specific choice:** twin tokens are paired only when both global crops show the same
sequence (`--cross-view-sequences same`), preserving this module's rule that cross-sequence
images are never patch-level targets. With `--cross-sequence-probability 0.25` that is ~75% of
samples; `any` would also pair co-registered voxels across contrasts (contrast-invariant patch
features, at the risk of suppressing lesions visible in only one sequence).

**Cross-sequence global pairs default to 0.25 (was 0.75).** A DINO target shared by a T1 and a
FLAIR crop rewards discarding contrast-specific findings (FLAIR-only gliosis, SWI-only bleeds),
and in an aligned space the cheapest shared signal is the crop's position. Many "different"
sequences are the same contrast in another plane, so the true cross-contrast share is lower still.
A small non-zero rate keeps one embedding space across contrasts for MIL bags that mix sequences;
0 vs 0.25 is to be settled with the validation MIL probe.

Production: `scripts/train_mrdino.sbatch` (ViT-H+, 4 nodes, 3 phases; see below).
`train_32n_7b.sbatch` / `train_32n_hplus.sbatch` keep the old recipe for reference only.

**Healthy early training (FORA reference, ViT-L, global batch 128–1024):** iBOT target top-1
rises from ~0.003 to >0.05 within 2.5k steps, patch target entropy falls, prototype usage stays
≳0.4, `cross_kl` ~1.5–2.5 without a rising trend. Gram anchoring (stage `gram`) gave no gain in
FORA and stays optional.

## Optional input-volume cache

From `contrastive-pretraining/`:

```bash
python scripts/preprocess_volumes.py \
  --data_folder /path/to/MR-RATE-coreg/mri \
  --out_dir /path/to/mrrate_preprocessed \
  --space coreg_space \
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

The unit/integration suite creates raw atlas-registered NIfTIs and an optional
dummy cache. It verifies that the raw transform is bit-for-bit the previous MIL
transform, plus manifest rejection, deterministic cross-sequence alignment,
complete sequence indexing, local/global containment, 3-D masks, a real
DINO+iBOT forward/backward optimizer step, EMA update, and checkpoint reload:

```bash
cd dino-pretraining
PYTHONPATH=/path/to/dinov3:$PYTHONPATH pytest -q
```

Run the actual four-GPU FSDP2 path on a generated dummy dataset:

```bash
sbatch scripts/smoke_dummy.sbatch
```

The job succeeds only after `step_00000002/COMPLETE` exists.

## Production training (3 phases, ViT-H+, 4 nodes)

`scripts/train_mrdino.sbatch` runs every phase. Set `STAGE`; everything else has a phase
default. It reads the extracted MR-RATE-coreg tree directly (`SPACE=atlas_space` switches to
the atlas tree).

**Model: ViT-H+ (0.84B).** This is the official DINOv3 H+ shape with 3-D patch embedding
and RoPE: 32 blocks, width 1280, 20 heads, SwiGLU ×6, 2×16×16 patch = 2×8×8 mm tokens. The
heads are 2048 → 256 with 16,384 cosine prototypes each.
- **Why not 7B:** the 7B (the previous MR default) failed in FORA with the old recipe, and
  4 nodes cannot train it.
- **Why not ViT-L:** ViT-L (0.30B, `ARCH=large`) is the size proven with this recipe in
  FORA CT, but it is smaller than wanted.
- **Learning rate:** it was tuned on ViT-L. Watch the target metrics in the first
  2–3k steps (see "Healthy early training" above).

| | Phase 1 `pretrain` | Phase 2 `gram` | Phase 3 `highres` |
|---|---|---|---|
| steps | **50,000** (~10 passes over ~635k sequences) | 10,000 | 5,000 |
| starts from | scratch | phase-1 end | phase-2 end (or phase-1 end, see below) |
| global crops | 2 × 64×192×192 vox (64×96×96 mm, 4,608 tokens) | same | 2 × 64×384×384 (whole axial plane, 18,432 tokens) |
| local crops | 8 × 32×96×96 | 8 × 32×96×96 | 4 × 32×192×192 |
| batch / GPU (global on 4 nodes) | 8 (128) | 8 (128) | 4 (64) |
| activation checkpointing | on (selective) | on | on |
| peak lr (at batch 1024, sqrt-scaled) | 1e-3 → 3.5e-4 | 5e-4 → 1.8e-4 | 4e-4 → 1e-4 |
| warmup / end lr | 2,500 / cosine to 1e-6 at step end | 250 / same | 500 / same |
| weight decay | 0.04 → 0.4 | 0.1 → 0.4 | 0.04 → 0.2 |
| teacher temperature | 0.04 → 0.07 over 10k | 0.07 fixed | 0.07 fixed |
| teacher EMA | 0.992 → 1 | 0.996 → 1 | 0.996 → 1 |
| prototype freeze | first 1,250 steps | none | none |
| Gram weight | 0 | 1.0 | 1.5 |

Phases 2 and 3 use FORA CT-DINO's tested settings (`phase2_gram_recipe.args`,
`phase3_hires_recipe.args`). A phase restarts its schedule at step 0, so reusing phase-1
values would redo the warmups: teacher temperature back to 0.04, prototypes frozen again.

**Settings shared by all phases:**
- **Losses:** DINO 1.0 + iBOT 1.0 (block masks 10–50%) + twin-token 1.0 + KoLeo 0.1.
  Sinkhorn targets use 2×2×2 spatial bins.
- **Sequences:** cross-sequence global pairs 25%, twin tokens within the same sequence.
- **Optimiser:** AdamW β2 0.999, clip 3, layer-wise lr decay 0.9, patch-embed lr ×0.2, BF16.
- **Infrastructure:** FSDP2, compiled blocks, checkpoint every 500 steps (last 6 kept).

**Memory and speed per H200** (1-node benchmark, H+, compiled, real crop sizes):

| Configuration | Result |
|---|---|
| phase-1 crops, batch 8, no checkpointing | **out of memory** |
| phase-1 crops, batch 8 + checkpointing | 0.39 steps/s, 32 GiB (default) |
| phase-1 crops, batch 4, no checkpointing | 0.95 steps/s, 83 GiB |
| highres crops, batch 2, no checkpointing | 128 GiB (too tight) |
| highres crops, batch 2 + checkpointing | 0.23 steps/s, 43 GiB |
| highres crops, batch 4 + checkpointing | 0.10 steps/s, 81 GiB (default) |

So on 4 nodes, as long as data loading keeps up, expect roughly:
- phase 1: ~36 h;
- phase 2: ~10 h (the Gram anchor adds a forward pass);
- phase 3: ~14 h.

Each phase is longer than one 24 h job; the launcher checkpoints at the wall-time signal
and requeues itself.

```bash
cd dino-pretraining
export DATA_FOLDER=/path/to/MR-RATE-coreg/mri SPLITS_CSV=/path/to/splits.csv
R=/path/to/mrdino3d_hplus

# phase 1 (+ validation probe of chosen steps, run alongside on the login node)
STAGE=pretrain OUTPUT=$R/pretrain sbatch scripts/train_mrdino.sbatch
nohup scripts/probe_watch.sh $R/pretrain 5000 10000 20000 30000 40000 50000 &

# phase 2 from the phase-1 end
STAGE=gram RESUME=$R/pretrain/checkpoints/step_00050000 OUTPUT=$R/gram sbatch scripts/train_mrdino.sbatch
nohup scripts/probe_watch.sh $R/gram 5000 10000 &

# phase 3 from the phase-2 end, only if the gram probe is >= the phase-1 probe;
# otherwise RESUME=$R/pretrain/checkpoints/step_00050000 (FORA CT: Gram gave no gain)
STAGE=highres RESUME=$R/gram/checkpoints/step_00010000 OUTPUT=$R/highres sbatch scripts/train_mrdino.sbatch
nohup scripts/probe_watch.sh $R/highres 2500 5000 &
```

**Rules:**
- **`RESUME` for phases 2/3:** it names the previous phase's checkpoint only for the first
  start. Once the phase has its own complete checkpoint, a requeued job continues from that
  one. A phase-2/3 launch without `RESUME` and without its own checkpoints is refused.
- **Resume does not cross spaces or arches:** checkpoints record their space, and the
  probe rebuilds the arch from the checkpoint metadata.
- **`STEPS` is the real run length:** the cosine decay ends there, so set it before
  launching, not mid-run.
- **Overrides:** `ARCH`, `STEPS`, `BATCH_SIZE`, `ACT_CKPT`, `GRAD_ACCUM_STEPS`, `WORKERS`,
  `LOCAL_CROPS`, `CROSS_SEQUENCE_PROBABILITY`, `CROSS_VIEW_SEQUENCES`, `GRAM_WEIGHT`,
  `RESUME`, `SPACE`, `COMPILE`. Container and paths: `SIF`, `DINOV3_ROOT`, `EXTRA_PIP`,
  `ZIG_ARCHIVE` (see `scripts/train_node.sh`).
- **Site-specific `#SBATCH` lines:** edit `--partition`, the `--exclude` bad-node list,
  and `--mail-user` before running elsewhere. Slurm refuses unknown node names.
- **Run `scripts/smoke_dummy.sbatch` first** (1 node, ~2 min) in the same container and
  DINOv3 checkout. It trains, resumes and runs the probe on synthetic coreg data.

`train_32n_7b.sbatch` and `train_32n_hplus.sbatch` are the old 7B-continuation recipe
(32-node DDP/FSDP) and are kept for reference only.

## Validation probe (checkpoint scoring)

This is the FORA CT-DINO protocol: frozen features, MIL on the validation split only, and
patient-level cross-validation. Train and test are never touched.

1. **Export** (`python -m mr_dino.probe export`): copies the EMA-teacher backbone out of a
   DCP checkpoint, so checkpoint rotation cannot delete it.
2. **Extract** (`python -m mr_dino.probe extract`, one worker per GPU):
   - Runs every sequence of every labelled `val` study (3,764 studies, 3,413 patients)
     through the training preprocessing of the checkpoint's own space/spacing/shape.
   - Cuts each volume into phase-1-size tiles (64×192×192 voxels) and keeps the last
     block's normalised patch tokens.
   - Average-pools the tokens 2×2×2 over head foreground and drops air.
   - Writes one fp16 token file plus one index per worker to node-local `/tmp`.
3. **CV** (`python -m mr_dino.probe cv --fold k`): trains MR-RATE's NeuroVFM
   `ClassifyThenAggregate` head per patient-level fold (5 folds, 8 epochs, batch 8,
   lr 2e-3, pos_weight clipped to [1, 100]). One bag holds all sequences of a study.
4. **Summarize**: out-of-fold macro and per-class AUROC/AUPRC → `results.json`,
   `per_class.csv`, `summary.txt`.

Labels default to the 14 merged neuroradiology groups
(`contrastive-pretraining/scripts/eval_labels/splits_merged_majority`). Set
`LABELS_CSV`/`SPLITS_CSV` to use the 32/37-pathology labels instead.

```bash
# one checkpoint, one node (extract -> 5 folds -> summary; ~1-2 h on 4 H200):
DATA_FOLDER=/path/to/MR-RATE-coreg/mri \
sbatch scripts/probe_checkpoint.sbatch /path/to/pretrain/checkpoints/step_00005000 /path/to/pretrain/probe/step_5000
# follow a running stage: export each listed step as it appears, then submit its probe:
DATA_FOLDER=/path/to/MR-RATE-coreg/mri nohup scripts/probe_watch.sh /path/to/pretrain 2500 5000 10000 15000 20000 &
# inside a running training allocation instead of a new job:
srun --overlap --jobid JOB -N1 -n1 -w NODE scripts/probe_node.sh CKPT OUT_DIR
```

Use the probe to:
- decide how long to run phase 1;
- check that phase 2 (Gram) does not lose AUROC before phase 3 starts from it;
- settle open recipe questions, e.g. `--cross-sequence-probability` 0 vs 0.25.

Compare per-class rows, not just the macro AUROC: FLAIR-, SWI- and DWI-dependent groups
are what cross-sequence choices can hurt.

## Lightweight/debug run

`mr_dino.train_ddp` supports one or more GPUs and a tiny backbone:

```bash
torchrun --standalone --nproc-per-node=1 -m mr_dino.train_ddp \
  --data-folder /path/to/MR-RATE-coreg/mri \
  --output-dir /tmp/mrdino_debug \
  --arch tiny --steps 10 --workers 0 --local-crops 2 \
  --prototypes 64 --head-hidden-dim 128 \
  --warmup-steps 1 --teacher-warmup-steps 1 \
  --save-every 5 --log-every 1 --resume none
```

This path is for correctness/debugging. Use FSDP2 for the 7B production model.

To use the optional NPZ input cache in a manual launch, replace
`--data-folder ...` with `--preprocessed-dir /path/to/mrrate_preprocessed`.
