# MR-RATE – Contrastive Pretraining Submodule

[![Tests](https://github.com/forithmus/MR-RATE/actions/workflows/tests.yml/badge.svg)](https://github.com/forithmus/MR-RATE/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/forithmus/MR-RATE/branch/main/graph/badge.svg)](https://codecov.io/gh/forithmus/MR-RATE)

Contrastive vision-language model for brain & spine MRI that aligns multi-sequence MRI volumes and radiology reports using VL-CABS loss. Each subject has a variable number of volumes (2-12+, e.g. T1w, T2w, FLAIR, SWI, DWI) which are fused via configurable strategies. Supports multiple vision encoder backbones, tiling strategies, and normalization methods.

## Architecture

Built on [FORA](https://github.com/forithmus/FORA), extending its image encoder and contrastive pretraining framework to multi-sequence brain & spine MRI with variable volumes per subject.

**Text Encoder**: [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized).

### Vision Encoders

| Encoder | `--encoder` | Backbone | Depth Handling | Description |
|---------|-------------|----------|---------------|-------------|
| VJEPA2 | `vjepa2` | [HuggingFace](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384) | Temporal CNN (4x stride) | ViT-G loaded via HuggingFace, depth compressed by CNN before transformer |
| VJEPA 2.1 | `vjepa21` | [torch.hub](https://github.com/facebookresearch/vjepa2) | Temporal CNN (4x stride) | ViT-G loaded via torch.hub, requires `.pt` checkpoint |
| VJEPA2 Sliding | `vjepa2_sliding` | HuggingFace | Tiled chunks + mean pool | Splits depth into non-overlapping chunks, processes independently, mean-pools |
| VJEPA 2.1 Sliding | `vjepa21_sliding` | torch.hub | Tiled chunks + mean pool | Same sliding approach with VJEPA 2.1 backbone |

**Temporal CNN** encoders use a `ResidualTemporalDownsample` module that compresses depth by 4x before the transformer sees it (256 slices → 64 frames).

**Sliding/Tiling** encoders skip the CNN entirely — they split the volume into `chunk_size` depth tiles (default: 64), encode each through the full transformer, and mean-pool the token outputs. During training, chunks are processed sequentially with gradient checkpointing and running mean pooling for memory efficiency. During inference, all chunks are batched for speed.

### Fusion Modes

For combining variable MRI volumes per subject:

| Mode | Strategy |
|------|----------|
| `early` | Stack volumes into channels before the encoder |
| `mid_cnn` | Process separately through CNN, merge features, then transformer |
| `late` | Siamese processing, merge at token level via masked average |
| `late_attn` | Siamese processing with learned attention-based pooling |

### Pooling Strategies

For `late_attn` mode:

| Strategy | Description |
|----------|-------------|
| `simple_attn` | Learned attention weights over volumes |
| `cross_attn` | Text-guided cross-attention pooling |
| `gated` | Gated attention with text conditioning |

### Data Preprocessing

All NIfTI volumes are reoriented to **canonical RAS** (via `nibabel.as_closest_canonical`) before processing, ensuring consistent axis orientation regardless of acquisition plane. Volumes are then:

1. **Resampled** to target spacing (default: 1.0mm axial, 0.5mm in-plane)
2. **Crop/padded** to fixed shape (default: 256×384×384) with a 15mm posterior shift on the Y axis to compensate for defacing
3. **Normalized** using configurable methods (z-score, percentile, min-max)

### Normalization Methods

| Method | Description |
|--------|-------------|
| `zscore` | Z-score on nonzero voxels, clip to [-5,5], rescale to [-1,1] |
| `percentile` | Clip to [0.5, 99.5] percentile, rescale to [-1,1] |
| `minmax` | Simple min-max rescale to [-1,1] |

### Offline Preprocessing (.npz cache)

The coregistered/atlas NIfTIs are large (some ~2 GB) and decoding + RAS-reorient +
resampling them on the fly starves the GPUs. `scripts/preprocess_volumes.py` runs the
**exact** per-volume pipeline above once, offline, and writes one compact `.npz` per
subject. Training/inference then read those directly with `--use_preprocessed`, skipping
the expensive NIfTI read entirely.

The script is independent of any dataloader/training decision: it discovers every subject
under `--data_folder` (report/split agnostic) and keys the output only by the preprocessing
config, which is recorded in a manifest that the loader validates.

**Output layout** (consumed by both `MRReportDataset` and `MRReportDatasetInfer`):

```
<out_dir>/<space>/_manifest.json        # space, spacing, shape, posterior shift, normalizer, dtype
<out_dir>/<space>/<study_uid>.npz       # volumes: [N, D, H, W]  (all of a subject's volumes, stacked)
```

The loader casts the cached array to bfloat16 and adds the channel dim → `[N, 1, D, H, W]`,
**byte-for-byte identical** to the live path (verified in `tests/test_preprocess_cache.py`).

**Step 1 — build the cache** (run once; resumable, shardable across array jobs):

```bash
python scripts/preprocess_volumes.py \
    --data_folder /path/to/MR-RATE-coreg/mri \
    --out_dir     /path/to/preprocessed \
    --space       coreg_space \
    --normalizer  zscore \
    --num_workers 8
# Shard across N jobs: add --num_shards N --shard_index $SLURM_ARRAY_TASK_ID
```

The script exposes **every** setting that affects the volume tensor, matching the
dataloader 1:1: `--space`, `--normalizer`, `--normalizer_kwargs` (JSON, e.g.
`'{"lower_percentile": 1.0, "upper_percentile": 99.0}'`), `--target_spacing`,
`--target_shape`, `--posterior_shift_mm`. All are recorded in the manifest and
checked at load time. Cache-only knobs: `--dtype {float16,float32}` (default
`float16`, ~2× smaller), `--compress` (smaller files, slower reads — default off
for fastest training reads). Orchestration: `--overwrite` (reprocess existing),
`--num_workers`, `--num_shards`/`--shard_index` (split across jobs/nodes), `--limit`.

> The preprocessing config passed here **must** match training (`--space`, `--normalizer`,
> and the spacing/shape/posterior-shift defaults). The manifest captures it and the
> dataloader refuses to train on a mismatched cache unless you pass `--cache_allow_mismatch`.

**Step 2 — train/infer from the cache** (drop `--data_folder`, add the two flags):

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa2 --fusion_mode late \
    --space coreg_space --normalizer zscore \
    --jsonl_file /path/to/reports.jsonl \
    --use_preprocessed \
    --preprocessed_dir /path/to/preprocessed
```

The same `--use_preprocessed` / `--preprocessed_dir` flags work for `inference.py` and
`extract_features.py`.

## Repository Structure

```
contrastive-pretraining/
├── mr_rate/                  # Core model package
│   ├── mr_rate/
│   │   └── mr_rate.py        # MRRATE model, pooling modules, VL-CABS loss
│   └── setup.py
├── vision_encoder/           # Vision encoder package
│   ├── vision_encoder/
│   │   ├── vjepa_encoder.py           # VJEPA2 with LoRA + temporal CNN
│   │   ├── vjepa21_encoder.py         # VJEPA 2.1 with LoRA + temporal CNN
│   │   ├── vjepa_sliding_encoder.py   # VJEPA2 sliding window (tiled depth chunks)
│   │   ├── vjepa21_sliding_encoder.py # VJEPA 2.1 sliding window
│   │   └── optimizer.py               # Optimizer utilities
│   └── setup.py
├── scripts/                  # Training, inference, and evaluation
│   ├── run_train.py          # Training entry point (all encoder variants)
│   ├── mr_rate_trainer.py    # Distributed trainer (accelerate, W&B, resume)
│   ├── data.py               # MR dataset with variable volumes per subject (live NIfTI or .npz cache)
│   ├── data_inference.py     # Inference dataset loader with optional labels (live NIfTI or .npz cache)
│   ├── preprocess_volumes.py # Offline: bake the per-volume pipeline into per-subject .npz (fast disk reads)
│   ├── inference.py          # Zero-shot brain MRI pathology classification (32 pathologies)
│   ├── extract_features.py   # Cache frozen encoder features per split (linear-probe step 1)
│   ├── linear_probe.py       # Train + evaluate a linear classifier on cached features (step 2)
│   ├── eval.py               # Evaluation metrics (AUROC, F1, bootstrap CIs)
│   ├── submit_train.sh       # SLURM submission script
│   ├── test_sliding.sh       # SLURM test: sliding window memory test
│   └── test_vjepa21.sh       # SLURM test: VJEPA 2.1 memory test
├── data/                     # Data files
│   ├── findings_sentences.jsonl  # Report sentences per subject (for training)
│   └── pathologies.json          # 32 SNOMED CT-grounded pathology definitions with pos/neg prompts
├── tests/                    # Unit tests (106 tests, 88% core coverage)
├── requirements.txt          # All dependencies
└── pyproject.toml            # Pytest + coverage configuration
```

## Installation

```bash
git clone https://github.com/forithmus/MR-RATE.git
cd MR-RATE/contrastive-pretraining

# Create environment
conda create -n mrrate python=3.11 -y
conda activate mrrate

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt

# Install local packages in editable mode
pip install -e mr_rate/ -e vision_encoder/
```

For VJEPA 2.1, the torch hub repo will be auto-downloaded on first use. You also need the pretrained checkpoint (`.pt` file).

## Data Format

Each subject directory contains MRI volumes organized in batch directories:

Two directory layouts are supported (auto-detected):

**Layout 1 — Space-based** (with `--space native_space`):
```
data_folder/
├── SUBJECT_001/
│   ├── native_space/
│   │   └── img/
│   │       ├── T1w.nii.gz
│   │       ├── T2w.nii.gz
│   │       ├── FLAIR.nii.gz
│   │       └── SWI.nii.gz      # Variable: 2-12+ volumes per subject
│   ├── atlas_space/
│   │   └── img/
│   │       └── ...
│   └── coreg_space/
│       └── img/
│           └── ...
├── SUBJECT_002/
│   └── ...
```

**Layout 2 — Batch-based** (HuggingFace dataset format):

The image subdirectory name is chosen automatically from `--space`:

| Space (`--space`) | Image subfolder | HuggingFace repo |
|-------------------|-----------------|------------------|
| `native_space` *(default)* | `img/` | [Forithmus/MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE) |
| `coreg_space` | `coreg_img/` | [Forithmus/MR-RATE-coreg](https://huggingface.co/datasets/Forithmus/MR-RATE-coreg) |
| `atlas_space` | `atlas_img/` | [Forithmus/MR-RATE-atlas](https://huggingface.co/datasets/Forithmus/MR-RATE-atlas) |

```
data_folder/                       # e.g. ./data/MR-RATE-coreg/mri
├── batch00/
│   ├── SUBJECT_001/
│   │   └── coreg_img/             # img/ for native, atlas_img/ for atlas
│   │       ├── SUBJECT_001_t1w-raw-axi.nii.gz
│   │       ├── SUBJECT_001_flair-raw-axi.nii.gz
│   │       └── ...
│   ├── SUBJECT_002/
│   │   └── coreg_img/
│   │       └── ...
├── batch01/
│   └── ...
```

After `merge_downloaded_repos.py` consolidates native + coreg + atlas under a single tree, all three subfolders (`img/`, `coreg_img/`, `atlas_img/`) live side-by-side under each study, and `--space` selects which one to load.

Reports are stored in a JSONL file with one entry per subject:

```json
{"volume_name": "SUBJECT_001", "valid_json": true, "extracted_sentences": ["Normal brain MRI.", "No acute infarction.", ...]}
```

## Training

MR-RATE uses [Accelerate](https://huggingface.co/docs/accelerate) for distributed training across multiple GPUs/nodes with SyncBatchNorm and mixed precision (bf16).

### Quick Start

```bash
# VJEPA2 with temporal CNN (default)
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa2 \
    --fusion_mode late \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl \
    --normalizer percentile

# VJEPA 2.1 with temporal CNN
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa21 \
    --vjepa21_checkpoint /path/to/vjepa2_1_vitg_384.pt \
    --fusion_mode late \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl

# VJEPA 2.1 with sliding window (tiled depth chunks)
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa21_sliding \
    --chunk_size 64 \
    --vjepa21_checkpoint /path/to/vjepa2_1_vitg_384.pt \
    --fusion_mode late \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl

# VJEPA2 with sliding window
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa2_sliding \
    --chunk_size 64 \
    --fusion_mode late \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl
```

### SLURM Cluster

```bash
# Submit with default settings
sbatch scripts/submit_train.sh

# Override parameters via environment variables
FUSION_MODE=late_attn POOLING_STRATEGY=cross_attn NUM_TRAIN_STEPS=50000 \
    sbatch scripts/submit_train.sh
```

### Training Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--encoder` | `vjepa2`, `vjepa21`, `vjepa2_sliding`, `vjepa21_sliding` | `vjepa2` | Vision encoder backbone |
| `--vjepa21_checkpoint` | — | — | Path to VJEPA 2.1 `.pt` weights (required for `vjepa21*`) |
| `--chunk_size` | — | `64` | Depth chunk size for sliding encoders (must be even) |
| `--fusion_mode` | `early`, `mid_cnn`, `late`, `late_attn` | `late` | How to combine multi-sequence MRI volumes |
| `--pooling_strategy` | `simple_attn`, `cross_attn`, `gated` | `simple_attn` | Volume pooling (used with `late_attn`) |
| `--data_folder` | — | required | Path to MR data folder |
| `--jsonl_file` | — | required | Path to reports JSONL file |
| `--space` | `native_space`, `coreg_space`, `atlas_space` | `native_space` | Selects the image subfolder under each study: `<space>/img/` (layout 1) or `img/` / `coreg_img/` / `atlas_img/` (layout 2) |
| `--normalizer` | `zscore`, `percentile`, `minmax` | `zscore` | Volume normalization method |
| `--use_preprocessed` | flag | off | Read precomputed `.npz` volumes instead of raw NIfTI (see [Offline Preprocessing](#offline-preprocessing-npz-cache)). Big I/O win for large coreg/atlas volumes. |
| `--preprocessed_dir` | — | — | Root of the `.npz` cache (from `preprocess_volumes.py`). Required with `--use_preprocessed`. `--data_folder` becomes optional. |
| `--cache_allow_mismatch` | flag | off | Downgrade a cache-manifest config mismatch from a hard error to a warning. |
| `--pathology_labels_csv` | — | — | Path to pathology labels CSV (e.g. `mrrate_labels.csv`). Required for rebalancing. |
| `--rebalance_strategy` | `inverse_freq`, `sqrt_inverse_freq`, `max_inverse_freq` | — (uniform) | Per-subject sampling weight strategy for upsampling rare pathologies. |
| `--rebalance_base_weight` | — | `1.0` | Base sampling weight for all-negative / unlabeled subjects. |
| `--num_train_steps` | — | `100001` | Total training steps |
| `--lr` | — | `1e-5` | Learning rate |
| `--results_folder` | — | `./mr_rate_results` | Checkpoint output directory |
| `--splits_csv` | — | — | Path to splits CSV (columns: batch_id, patient_uid, study_uid, split) |
| `--split` | `train`, `val`, `test` | `train` | Which split to use |
| `--resume` | flag | — | Resume from latest checkpoint in results_folder |
| `--wandb` | flag | — | Enable Weights & Biases logging |
| `--wandb_project` | — | `mr-rate` | W&B project name |
| `--wandb_run_name` | — | auto | W&B run name |
| `--pretrained_weights` | — | — | Path to pretrained weights to initialize from |

### Training Features

- **Checkpoint resume**: `--resume` finds the latest `MrRate.full.{step}.pt` checkpoint and continues training with optimizer/scheduler state preserved
- **W&B integration**: `--wandb` logs loss, learning rate, volume count per step; run ID is persisted for seamless resume across SLURM jobs
- **Gradient checkpointing**: Nested checkpointing at both volume level (in MRRATE) and chunk level (in sliding encoders) for maximum memory efficiency
- **Dual checkpoints**: Saves both model-only `.pt` (for inference) and full `.pt` (model + optimizer + scheduler + step, for resume)
- **Rare-pathology rebalancing** (see below): inverse-prevalence weighted sampling so contrastive batches see uncommon pathologies more often

### Rare-Pathology Rebalancing

Pathology prevalence in MR-RATE is heavily skewed (~43% of studies are all-negative; the rarest pathology, *Hemangioma of vertebral column*, occurs in ~0.23% of studies; *Gliosis* in ~37%). Under uniform sampling, contrastive batches are dominated by common findings and the model rarely sees rare ones. Passing `--pathology_labels_csv` together with `--rebalance_strategy` switches the training DataLoader to a `WeightedRandomSampler` whose per-subject weights are derived from inverse prevalence.

#### How the weighting works

Each subject's report is **multi-label** — a single scan can have zero, one, or many positive pathologies. Every subject is reduced to a single sampling weight `w_i`. Bigger weight = drawn more often. The three strategies differ in how a subject's binary label vector is collapsed into that one number.

Concrete walk-through with two pathologies and three subjects (using real prevalences from `mrrate_labels.csv`: *Hemangioma of vertebral column* ≈ 0.23% → inv-freq ≈ 433; *Gliosis* ≈ 37% → inv-freq ≈ 2.7):

| Subject | Hemangioma (rare) | Gliosis (common) | `inverse_freq` weight |
|---------|-------------------|------------------|-----------------------|
| A       | 1                 | 1                | `1 + 433 + 2.7 ≈ 436.7` |
| B       | 0                 | 1                | `1 +   0 + 2.7 ≈ 3.7`   |
| C       | 0                 | 0                | `1` (base)              |

Subject A is drawn ~436× more often than C, and ~118× more often than B.

#### Strategy choice

For each pathology `p`, `prevalence[p]` is the positive rate across labeled subjects in the active split, and `inv_freq[p] = 1 / max(prevalence[p], eps)`. The three strategies are different ways to aggregate across a subject's positive labels:

| Strategy | Formula | Subject A's weight | Behavior |
|----------|---------|--------------------|----------|
| `inverse_freq` | `base + Σ_p y_p · inv_freq[p]` | `436.7` | **Sum** — rewards multiple co-occurring rare findings. On `mrrate_labels.csv`, rarest pathology mass: 0.23% → ~4.0% (~17× boost). |
| `sqrt_inverse_freq` | `base + Σ_p y_p · √inv_freq[p]` | `22.4` | Sum with diminishing returns. Rarest pathology mass: 0.23% → ~1.5%. Use when `inverse_freq` over-amplifies. |
| `max_inverse_freq` | `max(base, max_p y_p · inv_freq[p])` | `433` | **Max** — a subject's weight is set by its single rarest positive pathology. Doesn't matter if they have one rare finding or five — caps combinatorial blow-up. |

Recommendation: start with `inverse_freq`. Switch to `sqrt_inverse_freq` if training loss gets noisy from over-sampling the same handful of rare subjects.

Subjects missing from the labels CSV — and all-negative subjects — receive `--rebalance_base_weight` (default `1.0`), so they remain in the pool but do not dominate it.

#### Enabling it

Add two flags to your existing training command:

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa2 --fusion_mode late \
    --data_folder /path/to/data \
    --jsonl_file  /path/to/reports.jsonl \
    --pathology_labels_csv /path/to/mrrate_labels.csv \
    --rebalance_strategy inverse_freq
```

Without those flags, nothing changes — uniform `shuffle=True` as before. With them, the dataset prints a one-line summary at startup so you can verify the math:

```
[MRReportDataset] Rebalancing enabled (strategy=inverse_freq): 97896/97896 subjects matched labels CSV, weight range=[1.0, 1418.9], mean=33.0
[Trainer] Using WeightedRandomSampler (strategy=inverse_freq)
```

Total compute is unchanged (still `--num_train_steps` steps × batch_size × num_processes draws). Rebalancing only **redistributes** that draw budget toward rare-pathology subjects via `WeightedRandomSampler(..., replacement=True)`.

### Training Configuration

Key parameters in `scripts/run_train.py`:

```python
# Model
dim_text = 768          # BiomedVLP-CXR-BERT hidden size
dim_latent = 512        # Shared latent dimension
lora_r = 32             # LoRA rank
lora_alpha = 64         # LoRA alpha

# Training
batch_size = 1          # Per-GPU (each subject has variable volumes)
lr = 1e-5               # Learning rate
warmup_steps = 500      # Linear warmup steps
num_train_steps = 100001
save_model_every = 500  # Checkpoint frequency
```

## Inference

Zero-shot brain MRI pathology classification using text-guided similarity scoring against 32 SNOMED CT-grounded brain/spine MRI pathologies.

For each pathology, the model computes:

```
score = similarity("There is {finding}") − similarity("There is no {finding}")
```

using the positive/negative prompt pairs defined in [`data/pathologies.json`](data/pathologies.json). Positive scores indicate predicted presence.

### Pathologies

32 pathologies covering brain and spine MRI findings, each grounded to SNOMED CT or RadLex. Prompt pairs are tuned to match radiological language (e.g., "There is infarct" rather than "There is cerebral infarction"). See [`data/pathologies.json`](data/pathologies.json) for the full list with positive/negative prompts.

The shipped 32-class list drops 5 categories from the original 37 (`Ventriculomegaly`, `Cerebral infarction`, `Mastoiditis`, `Cerebral hemorrhage`, `Lipoma of brain`) because the Claude/GPT auto-labeling pipeline could not produce reliable agreement on them. The label CSVs distributed alongside MR-RATE use the same 32 columns in the same order, so prompt index ↔ label column ↔ output index are aligned across `inference.py`, `extract_features.py`, and `linear_probe.py`.

### Basic Usage

```bash
python scripts/inference.py \
    --encoder vjepa2 \
    --fusion_mode late \
    --pooling_strategy simple_attn \
    --weights_path ./mr_rate_results/MrRate.5000.pt \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl \
    --pathologies_file data/pathologies.json \
    --labels_file /path/to/labels.csv \
    --normalizer zscore \
    --results_folder ./inference_results

# With split filtering
python scripts/inference.py \
    --encoder vjepa2 \
    --fusion_mode late \
    --pooling_strategy simple_attn \
    --weights_path ./mr_rate_results/MrRate.5000.pt \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl \
    --pathologies_file data/pathologies.json \
    --labels_file /path/to/labels.csv \
    --splits_csv /path/to/splits.csv \
    --split test \
    --results_folder ./inference_results
```

### Pathologies File Format

The `--pathologies_file` argument supports two formats:

**Format 1** — Structured (recommended, matches `data/pathologies.json`):
```json
{
  "pathologies": {
    "Cerebral infarction": {
      "positive": "There is infarct",
      "negative": "There is no infarct"
    },
    "Glioma": {
      "positive": "There is glioma",
      "negative": "There is no glioma"
    }
  }
}
```

**Format 2** — Simple list (legacy, uses generic "There is {name}" prompts):
```json
["Cerebral infarction", "Glioma", "Pituitary adenoma"]
```

### Labels CSV Format

The `--labels_file` CSV contains binary (0/1) pathology labels per study, used for AUROC evaluation after inference. Expected format:

```csv
study_uid,Cerebral infarction,Cerebral hemorrhage,...,Glioma,Pituitary adenoma
STUDY001,0,1,...,0,1
STUDY002,1,0,...,1,0
```

Column names must match the pathology labels in `--pathologies_file`. The pre-computed labels for MR-RATE (97,896 studies, 32 pathologies) are available in the [Forithmus/MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE) HuggingFace repository at [`pathology_labels/mrrate_labels.csv`](https://huggingface.co/datasets/Forithmus/MR-RATE/blob/main/pathology_labels/mrrate_labels.csv). Labels are generated by the [pathology classification pipeline](../data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/06_pathology_classification/).

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder` | `vjepa2` | Must match the encoder used during training |
| `--fusion_mode` | required | Fusion mode used during training |
| `--pooling_strategy` | `simple_attn` | Pooling strategy used during training |
| `--weights_path` | required | Path to model checkpoint |
| `--data_folder` | required* | Path to MR data folder (*optional if `--use_preprocessed`) |
| `--jsonl_file` | required | Path to reports JSONL file |
| `--normalizer` | `zscore` | Volume normalization method |
| `--use_preprocessed` | off | Read precomputed `.npz` instead of raw NIfTI (see [Offline Preprocessing](#offline-preprocessing-npz-cache)) |
| `--preprocessed_dir` | — | Root of the `.npz` cache; required with `--use_preprocessed` |
| `--cache_allow_mismatch` | off | Downgrade a cache-manifest config mismatch to a warning |
| `--batch_size` | `1` | Inference batch size |
| `--results_folder` | `./inference_results` | Output directory |
| `--labels_file` | required | Path to labels CSV (study_uid + binary pathology columns) |
| `--splits_csv` | — | Path to splits CSV (columns: study_uid, split) |
| `--split` | `test` | Which split to evaluate |
| `--pathologies_file` | required | Pathologies JSON with positive/negative prompts (see format above) |

> `extract_features.py` accepts the same `--use_preprocessed` / `--preprocessed_dir` /
> `--cache_allow_mismatch` flags, so linear-probe feature caching can read from the
> `.npz` cache too.

### Outputs

| File | Content |
|------|---------|
| `predicted_scores.npz` | Raw prediction scores per pathology |
| `labels.npz` | Ground truth labels (if provided) |
| `subject_ids.txt` | Subject IDs processed |
| `scores.json` | Per-subject scores in JSON format |
| `aurocs.xlsx` | Per-pathology AUROC scores (if labels provided) |

## Linear Probing

A supervised baseline that freezes the pretrained MR-RATE encoder and trains only a linear classifier on top of its pooled visual features. This is the standard probe used to measure how well the contrastive representation captures the downstream pathology labels, separate from the zero-shot prompt quality.

The probe is two (optionally three) scripts run in order:

1. **`extract_features.py`** — encode every subject once with the frozen encoder and dump features to disk. This is the slow step; it runs the same model + preprocessing path as `inference.py`. Run it once per split.
2. **`linear_probe.py`** — train `nn.Linear(dim_latent, num_classes)` with `BCEWithLogitsLoss` on the cached features, pick the best epoch by val mean-AUROC, report test metrics through `eval.evaluate_internal` (identical AUROC pipeline to inference).
3. **`relabel_features.py`** *(optional)* — swap in a different label set on top of already-cached features **without re-encoding**. See [Reusing features for a different label set](#reusing-features-for-a-different-label-set-no-re-extract).

Why precompute features: the 3D encoder is the expensive part. Once it is frozen, every training epoch sees the same features, so running the encoder once and then training the linear head on cached `.npy` files is orders of magnitude faster than re-encoding each epoch — the standard CLIP / SimCLR / DINO linear-probe recipe. The corollary: **the labels are not part of the features**, so changing the label set never requires re-extraction — only the cheap `linear_probe.py` step is repeated (see step 3).

### Label sets

The class count is **derived from the labels CSV at runtime** — `extract_features.py` sets `num_classes = len(label_columns)` and writes `label_names.json` from the CSV header, and `linear_probe.py` sizes `nn.Linear(dim, n_classes)` from that. Nothing is hardcoded, so any labels CSV (`study_uid` + binary class columns) is a drop-in `--labels_file` with no code change. The ready-made set lives under `scripts/eval_labels/`:

| Folder | Classes | Ground-truth rule |
|--------|--------:|-------------------|
| `splits_merged_majority/` (`mrrate_merged_labels.csv`) | 14 | 3-model majority (Claude Opus 4.7 + GPT-5.5 + Nemotron-3 Super 120B; positive when ≥2 of the available votes agree), then collapsed into the neuroradiologist's clinical groups (8 pathophysiology `PP_*` + 6 imaging-phenotype `BP_*`) |

It pairs a labels CSV (`study_uid` + 14 binary class columns) with a `splits.csv` (`study_uid,split`). Reproduce it with `scripts/eval_labels/build_merged_group_labels.py --source majority` (the script also supports `--source {raw,csv32}` for 2-model agreement variants if you need them).

### Step 1 — Cache features

```bash
for SPLIT in train val test; do
  python scripts/extract_features.py \
      --encoder vjepa2 \
      --fusion_mode late \
      --pooling_strategy simple_attn \
      --weights_path ./mr_rate_results/MrRate.5000.pt \
      --data_folder /path/to/data \
      --jsonl_file /path/to/reports.jsonl \
      --labels_file /path/to/mrrate_labels.csv \
      --splits_csv /path/to/splits.csv \
      --split $SPLIT \
      --normalizer zscore \
      --out_dir ./linear_probe_features
done
```

For each split this writes:

| File | Shape | Content |
|------|-------|---------|
| `features_<split>.npy` | `[N, dim_latent]` (float32) | Frozen encoder output (L2-normalised masked-mean pool) |
| `labels_<split>.npy`   | `[N, num_classes]` (float32) | 0/1 ground-truth labels |
| `subject_ids_<split>.txt` | `N` lines | One `study_uid` per line, same order as the `.npy` rows |
| `label_names.json` | — | Pathology names; written once on the first split |

The script verifies the checkpoint actually loaded into the architecture: it hashes a sampled set of weights before and after `clip.load(...)` and aborts with a clear error if zero parameters changed (catches a wrong `--fusion_mode` / `--encoder` / `--dim_latent` that `strict=False` would otherwise silently swallow). It also prints any missing/unexpected state-dict keys; pass `--strict_missing` to fail on any missing key.

### Step 2 — Train and evaluate the linear head

```bash
python scripts/linear_probe.py \
    --features_dir ./linear_probe_features \
    --results_dir ./linear_probe_results \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-3 \
    --weight_decay 1e-4
```

Class-imbalance handling: pathologies are very rare (median prevalence ~1%), so the head trains with per-class `pos_weight = #neg / #pos` (capped at 100) computed on the training split. Pass `--no_pos_weight` to disable.

Outputs in `--results_dir`:

| File | Content |
|------|---------|
| `linear_head.pt` | Trained linear weights + label names + args |
| `history.json` | Per-epoch train loss and val mean AUROC |
| `per_class_test_auroc.json` | Final mean and per-class test AUROC |
| `test_logits.npy` / `test_labels.npy` / `test_subject_ids.txt` | Test-set predictions for downstream analysis |
| `test_aurocs.csv` | Wide-format per-class AUROC table |

The best epoch by val mean-AUROC is restored before test evaluation. Single-class columns (all 0 or all 1 in the test split) are gracefully reported as `NaN` and excluded from the macro mean.

### Reusing features for a different label set (no re-extract)

Because labels are not baked into the encoder features, you only ever run `extract_features.py` **once**. To probe a different label CSV — a different ground-truth rule, or a different grouping — relabel the cached features instead of re-encoding:

```bash
# (once) cache features
for SPLIT in train val test; do
  python scripts/extract_features.py \
      --encoder vjepa2 --fusion_mode late --pooling_strategy simple_attn \
      --weights_path ./mr_rate_results/MrRate.5000.pt \
      --data_folder /path/to/data --jsonl_file /path/to/reports.jsonl \
      --labels_file scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv \
      --splits_csv  scripts/eval_labels/splits_merged_majority/splits.csv \
      --split $SPLIT --normalizer zscore --out_dir ./lp_features
done

# train the head on those features
python scripts/linear_probe.py --features_dir ./lp_features --results_dir ./lp_results

# later: probe ANY other labels CSV on the SAME features — instant, no encoder pass
python scripts/relabel_features.py \
    --features_dir ./lp_features \
    --labels_file  /path/to/other_labels.csv \
    --out_dir      ./lp_features_other
python scripts/linear_probe.py --features_dir ./lp_features_other --results_dir ./lp_results_other
```

`relabel_features.py` symlinks `features_<split>.npy` + `subject_ids_<split>.txt` from `--features_dir` and rebuilds only `labels_<split>.npy` (in the exact subject order of the source) and `label_names.json` from `--labels_file`. If a cached subject is missing from the new labels CSV it **errors out** rather than silently misaligning rows. Use `--copy` to copy the feature files instead of symlinking (e.g. to move the dir to another machine).

### Notes

- Both scripts are single-process. The linear head is tiny — `Linear(512, num_classes)`, e.g. ~16K params for 32 classes or ~7K for the 14 merged groups — over ~180 MB of cached features, so DDP overhead would dominate. For feature extraction, shard externally by running one job per split, or by splitting `splits.csv` into chunks and concatenating the resulting `.npy` files.
- `--encoder`, `--fusion_mode`, `--pooling_strategy`, and `--dim_latent` must match the values used when the checkpoint was trained; if they don't, `_load_and_verify` will surface it loudly instead of silently loading garbage.

## Testing

```bash
# Run all tests with coverage
python -m pytest

# Run specific test files
python -m pytest tests/test_mr_rate_model.py -v
python -m pytest tests/test_fusion_modes.py -v

# Run without coverage (faster)
python -m pytest --no-cov
```

### Test Suite (106 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `test_imports.py` | Dependency + package import verification | All imports |
| `test_data.py` | Normalizers (zscore, percentile, minmax), collate_fn | Data pipeline |
| `test_preprocess_cache.py` | discover_subjects, preprocess_volumes.py, live↔cache equivalence, manifest guard | `.npz` cache |
| `test_pooling.py` | SimpleAttnPool, CrossAttnPool, GatedAttnPool | Shapes, masking, gradients |
| `test_mr_rate_model.py` | MRRATE model init, forward, loss, serialization | 95% of core model |
| `test_fusion_modes.py` | All 4 fusion modes x all pooling strategies | End-to-end forward pass |
| `test_vision_encoder.py` | ResidualTemporalDownsample, VJEPA2 preprocessing | CNN shapes, gradients |
