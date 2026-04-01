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
│   ├── data.py               # MR dataset with variable volumes per subject
│   ├── data_inference.py     # Inference dataset loader with optional labels
│   ├── inference.py          # Zero-shot brain MRI pathology classification (37 pathologies)
│   ├── eval.py               # Evaluation metrics (AUROC, F1, bootstrap CIs)
│   ├── submit_train.sh       # SLURM submission script
│   ├── test_sliding.sh       # SLURM test: sliding window memory test
│   └── test_vjepa21.sh       # SLURM test: VJEPA 2.1 memory test
├── data/                     # Data files
│   ├── findings_sentences.jsonl  # Report sentences per subject (for training)
│   └── pathologies.json          # 37 SNOMED CT-grounded pathology definitions with pos/neg prompts
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
```
data_folder/
├── batch00/
│   ├── SUBJECT_001/
│   │   └── img/
│   │       ├── SUBJECT_001_t1w-raw-axi.nii.gz
│   │       ├── SUBJECT_001_flair-raw-axi.nii.gz
│   │       └── ...
│   ├── SUBJECT_002/
│   │   └── img/
│   │       └── ...
├── batch01/
│   └── ...
```

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
| `--space` | — | `native_space` | Image space subfolder (only for space-based layout) |
| `--normalizer` | `zscore`, `percentile`, `minmax` | `zscore` | Volume normalization method |
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

Zero-shot brain MRI pathology classification using text-guided similarity scoring against 37 SNOMED CT-grounded brain/spine MRI pathologies.

For each pathology, the model computes:

```
score = similarity("There is {finding}") − similarity("There is no {finding}")
```

using the positive/negative prompt pairs defined in [`data/pathologies.json`](data/pathologies.json). Positive scores indicate predicted presence.

### Pathologies

37 pathologies covering brain and spine MRI findings, each grounded to SNOMED CT or RadLex. Prompt pairs are tuned to match radiological language (e.g., "There is infarct" rather than "There is cerebral infarction"). See [`data/pathologies.json`](data/pathologies.json) for the full list with positive/negative prompts.

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

Column names must match the pathology labels in `--pathologies_file`. The pre-computed labels for MR-RATE (97,896 studies, 37 pathologies) are available in the [Forithmus/MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE) HuggingFace repository at [`pathology_labels/mrrate_labels.csv`](https://huggingface.co/datasets/Forithmus/MR-RATE/blob/main/pathology_labels/mrrate_labels.csv). Labels are generated by the [pathology classification pipeline](../data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/06_pathology_classification/).

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder` | `vjepa2` | Must match the encoder used during training |
| `--fusion_mode` | required | Fusion mode used during training |
| `--pooling_strategy` | `simple_attn` | Pooling strategy used during training |
| `--weights_path` | required | Path to model checkpoint |
| `--data_folder` | required | Path to MR data folder |
| `--jsonl_file` | required | Path to reports JSONL file |
| `--normalizer` | `zscore` | Volume normalization method |
| `--batch_size` | `1` | Inference batch size |
| `--results_folder` | `./inference_results` | Output directory |
| `--labels_file` | required | Path to labels CSV (study_uid + binary pathology columns) |
| `--splits_csv` | — | Path to splits CSV (columns: study_uid, split) |
| `--split` | `test` | Which split to evaluate |
| `--pathologies_file` | required | Pathologies JSON with positive/negative prompts (see format above) |

### Outputs

| File | Content |
|------|---------|
| `predicted_scores.npz` | Raw prediction scores per pathology |
| `labels.npz` | Ground truth labels (if provided) |
| `subject_ids.txt` | Subject IDs processed |
| `scores.json` | Per-subject scores in JSON format |
| `aurocs.xlsx` | Per-pathology AUROC scores (if labels provided) |

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
| `test_pooling.py` | SimpleAttnPool, CrossAttnPool, GatedAttnPool | Shapes, masking, gradients |
| `test_mr_rate_model.py` | MRRATE model init, forward, loss, serialization | 95% of core model |
| `test_fusion_modes.py` | All 4 fusion modes x all pooling strategies | End-to-end forward pass |
| `test_vision_encoder.py` | ResidualTemporalDownsample, VJEPA2 preprocessing | CNN shapes, gradients |
