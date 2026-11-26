# MIL ablations

Three ablations of the MR-RATE MIL classifier (ViT-H+ DINO backbone, 74 labels). All use the
resumable multi-node trainer `dfw_mil.mil_fast` and the launcher `scripts/dfw/train_array.sbatch`,
with the baseline settings (head lr 0.002, last-block lr 1e-5, 10 epochs, one study per rank per
step, validation every epoch on the full validation split).

Set `CODE` to the `dino-pretraining` folder of this checkout. `mr_dino` and `dfw_mil` are both
imported from it.

## 1. Training-data size

`--train-fraction f` keeps a fraction `f` of the training patients. Patients are ordered once by a
hash of their id and `--subset-seed`, and a fraction keeps the first `ceil(f * n)` of them, so the
20% subset is contained in the 40% subset and so on, and all studies of a patient stay together.
Validation and test are never subsetted. The feature normalization is computed on the same subset.

```bash
for f in 0.2 0.4 0.6 0.8; do TRAIN_FRACTION=$f sbatch scripts/dfw/train_array.sbatch; done
```

Runs go to `runs/mil_frac<f>`. The 100% point is the baseline run.

## 2. Cross-sequence attention

`--fusion cross_sequence` inserts `dfw_mil.fusion.CrossSequenceAttention` between the trainable
last block and the MIL head. All sequences of a study share the atlas grid, so at every token
position the tokens of the sequences covering it (T1, T2, FLAIR, SWI and further acquisitions)
attend to each other, with a learned embedding of each token's sequence kind. Every sequence
token stays in the bag. The output projections start at zero, so the untrained model equals the
baseline. The module is deterministic, which keeps the two-pass MIL gradient exact, and the last
block is recomputed per sequence in the backward pass (`--fusion-recompute`) to bound memory.

```bash
FUSION=cross_sequence NORMALIZATION=<baseline run>/normalization.pt sbatch scripts/dfw/train_array.sbatch
```

Options: `--fusion-depth` (1), `--fusion-heads` (8), `--fusion-mlp-ratio` (2), `FUSION_LR` (1e-4).
The baseline normalization can be reused because the backbone and training set are unchanged.

## 3. Modalities as input channels

The backbone takes T1, T2, FLAIR and SWI as the channels of one input, plus one availability
plane per modality (8 input channels, `mr_dino.channels`). Missing modalities are zero with
availability 0. Contrast-enhanced T1 uses the T1 channel. Other sequences are not used.

**DINO pretraining** (`--input-channels modality`, raw NIfTI input only). One item per study.
Each item draws one acquisition per modality. Each view keeps a random nonempty subset of the
available modalities, the full set with probability `--full-modality-probability` (0.5). The two
global views share their subset unless, with `--cross-sequence-probability`, the second draws its
own, and iBOT/cross-crop twin tokens are paired only between views with the same subset. Add
`--input-channels modality` to the trainer arguments of each stage (pretrain, gram, highres) and
keep every other setting of the baseline recipe.

**Export** the final teacher as a release:

```bash
python -m mr_dino.export_release --checkpoint <highres run>/checkpoints/step_00005000 --output <release dir>
```

The release records `in_chans: 8` and its channel layout in `config.json`.

**MIL** with that release. Each study is a bag of modality stacks. Stack `k` uses acquisition
`k mod n_m` of every modality `m`, so every acquisition enters the bag at least once.

```bash
RELEASE=<release dir> RUN=<new run dir> sbatch scripts/dfw/train_array.sbatch
```

## Smoke test

`SMOKE=1` trains 2 epochs on 96 training and 24 validation studies, checkpoints after 3 steps and
resumes in the same job:

```bash
SMOKE=1 sbatch --array=1 --time=01:30:00 scripts/dfw/train_array.sbatch
SMOKE=1 FUSION=cross_sequence sbatch --array=1 --time=01:30:00 scripts/dfw/train_array.sbatch
```

## Tests

```bash
python -m pytest tests/test_ablations.py
```
