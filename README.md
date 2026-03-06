# MR-RATE

[![Tests](https://github.com/forithmus/MR-RATE/actions/workflows/tests.yml/badge.svg)](https://github.com/forithmus/MR-RATE/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/forithmus/MR-RATE/branch/main/graph/badge.svg)](https://codecov.io/gh/forithmus/MR-RATE)

Contrastive vision-language model for brain MRI that aligns multi-sequence MRI volumes and radiology reports using VL-CABS loss. Each subject has a variable number of volumes (2-12+, e.g. T1w, T2w, FLAIR, SWI, DWI) which are fused via configurable strategies. Supports multiple image spaces and normalization methods.

## Architecture

**Image Encoder**: [VJEPA2](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384) (ViT-G) with LoRA fine-tuning and a temporal CNN for depth downsampling.

**Text Encoder**: [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized).

**Fusion Modes** for combining variable MRI volumes per subject:

| Mode | Strategy |
|------|----------|
| `early` | Stack volumes into channels before the encoder |
| `mid_cnn` | Process separately through CNN, merge features, then transformer |
| `late` | Siamese processing, merge at token level via masked average |
| `late_attn` | Siamese processing with learned attention-based pooling |

**Pooling Strategies** (for `late_attn` mode):

| Strategy | Description |
|----------|-------------|
| `simple_attn` | Learned attention weights over volumes |
| `cross_attn` | Text-guided cross-attention pooling |
| `gated` | Gated attention with text conditioning |

**Image Spaces**: Data can be organized in different spatial registrations:

| Space | Description |
|-------|-------------|
| `native_space` | Original acquisition space |
| `atlas_space` | Registered to standard brain atlas |
| `coreg_space` | Co-registered across sequences |

**Normalization Methods**:

| Method | Description |
|--------|-------------|
| `zscore` | Z-score on nonzero voxels, clip to [-5,5], rescale to [-1,1] |
| `percentile` | Clip to [0.5, 99.5] percentile, rescale to [-1,1] |
| `minmax` | Simple min-max rescale to [-1,1] |

## Repository Structure

```
MR-RATE/
├── mr_rate/                  # Core model package
│   ├── mr_rate/
│   │   └── mr_rate.py        # MRRATE model, pooling modules, VL-CABS loss
│   └── setup.py
├── vision_encoder/           # Vision encoder package
│   ├── vision_encoder/
│   │   ├── vjepa_encoder.py  # VJEPA2Encoder with LoRA + temporal CNN
│   │   └── optimizer.py      # Optimizer utilities
│   └── setup.py
├── scripts/                  # Training, inference, and evaluation
│   ├── run_train.py          # Training entry point
│   ├── mr_rate_trainer.py    # Distributed trainer (accelerate, W&B, resume)
│   ├── data.py               # MR dataset with variable volumes per subject
│   ├── data_inference.py     # Inference dataset loader
│   ├── inference.py          # Zero-shot brain MRI pathology classification
│   ├── eval.py               # Evaluation metrics (AUROC, F1, bootstrap CIs)
│   └── submit_train.sh       # SLURM submission script
├── tests/                    # Unit tests (106 tests, 88% core coverage)
├── requirements.txt          # All dependencies
└── pyproject.toml            # Pytest + coverage configuration
```

## Installation

```bash
git clone https://github.com/forithmus/MR-RATE.git
cd MR-RATE

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install local packages in editable mode
pip install -e mr_rate/ -e vision_encoder/
```

## Data Format

Each subject directory contains MRI volumes organized by spatial registration:

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

Reports are stored in a JSONL file with one entry per subject:

```json
{"subject_id": "SUBJECT_001", "valid_json": true, "extracted_sentences": ["Normal brain MRI.", "No acute infarction.", ...]}
```

## Training

MR-RATE uses [Accelerate](https://huggingface.co/docs/accelerate) for distributed training across multiple GPUs/nodes with SyncBatchNorm and mixed precision (bf16).

### Quick Start

```bash
# Single-node multi-GPU
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --fusion_mode late \
    --pooling_strategy simple_attn \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl \
    --space native_space \
    --normalizer zscore
```

### Multi-Node Distributed Training

```bash
accelerate launch \
    --multi_gpu \
    --num_machines <NUM_NODES> \
    --num_processes <TOTAL_GPUS> \
    --mixed_precision bf16 \
    --machine_rank <NODE_RANK> \
    --main_process_ip <MASTER_ADDR> \
    --main_process_port <MASTER_PORT> \
    scripts/run_train.py \
        --fusion_mode late \
        --pooling_strategy simple_attn \
        --data_folder /path/to/data \
        --jsonl_file /path/to/reports.jsonl
```

### SLURM Cluster

A ready-to-use SLURM script is provided at `scripts/submit_train.sh`:

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
| `--fusion_mode` | `early`, `mid_cnn`, `late`, `late_attn` | `late` | How to combine multi-sequence MRI volumes |
| `--pooling_strategy` | `simple_attn`, `cross_attn`, `gated` | `simple_attn` | Volume pooling (used with `late_attn`) |
| `--data_folder` | — | required | Path to MR data folder |
| `--jsonl_file` | — | required | Path to reports JSONL file |
| `--space` | `native_space`, `atlas_space`, `coreg_space` | `native_space` | Image space subfolder |
| `--normalizer` | `zscore`, `percentile`, `minmax` | `zscore` | Volume normalization method |
| `--num_train_steps` | — | `100001` | Total training steps |
| `--lr` | — | `1e-5` | Learning rate |
| `--results_folder` | — | `./mr_rate_results` | Checkpoint output directory |
| `--resume` | flag | — | Resume from latest checkpoint in results_folder |
| `--wandb` | flag | — | Enable Weights & Biases logging |
| `--wandb_project` | — | `mr-rate` | W&B project name |
| `--wandb_run_name` | — | auto | W&B run name |
| `--pretrained_weights` | — | — | Path to pretrained weights to initialize from |

### Training Features

- **Checkpoint resume**: `--resume` finds the latest `MrRate.full.{step}.pt` checkpoint and continues training with optimizer/scheduler state preserved
- **W&B integration**: `--wandb` logs loss, learning rate, volume count per step; run ID is persisted for seamless resume across SLURM jobs
- **DDP volume padding**: Subjects with different volume counts are automatically padded across ranks to keep SyncBatchNorm synchronized; padded volumes are masked out during pooling
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

Zero-shot brain MRI pathology classification using text-guided similarity scoring.

### Basic Usage

```bash
python scripts/inference.py \
    --fusion_mode late \
    --pooling_strategy simple_attn \
    --weights_path ./mr_rate_results/MrRate.5000.pt \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl \
    --space native_space \
    --normalizer zscore \
    --results_folder ./inference_results
```

### With Evaluation

```bash
python scripts/inference.py \
    --fusion_mode late \
    --weights_path ./mr_rate_results/MrRate.5000.pt \
    --data_folder /path/to/data \
    --jsonl_file /path/to/reports.jsonl \
    --labels_file /path/to/labels.csv \
    --results_folder ./inference_results
```

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fusion_mode` | required | Fusion mode used during training |
| `--pooling_strategy` | `simple_attn` | Pooling strategy used during training |
| `--weights_path` | required | Path to model checkpoint |
| `--data_folder` | required | Path to MR data folder |
| `--jsonl_file` | required | Path to reports JSONL file |
| `--space` | `native_space` | Image space subfolder |
| `--normalizer` | `zscore` | Volume normalization method |
| `--batch_size` | `1` | Inference batch size |
| `--results_folder` | `./inference_results` | Output directory |
| `--labels_file` | — | Path to labels CSV (enables AUROC evaluation) |
| `--pathologies_file` | — | JSON file with custom pathology names |

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

## License

We are committed to fostering innovation and collaboration in the research community. All elements of the MR-RATE repository are released under the **Creative Commons Attribution–NonCommercial–ShareAlike (CC BY-NC-SA)** license.

This allows all elements to be freely used, modified, and shared for **non-commercial research purposes**, provided that the original work is properly cited and any derivative works are distributed under the same license.
