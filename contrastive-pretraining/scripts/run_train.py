import argparse
from transformers import BertTokenizer, BertModel
from mr_rate import MRRATE
from mr_rate_trainer import MrRateTrainer
from vision_encoder import VJEPA2Encoder
import torch
import torch.nn as nn


def convert_bn_to_syncbn(module):
    """Convert all BatchNorm layers to SyncBatchNorm for proper distributed training."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(module)


parser = argparse.ArgumentParser(description='MR-RATE Training')
parser.add_argument('--fusion_mode', type=str, default='late',
                    choices=['early', 'mid_cnn', 'late', 'late_attn'])
parser.add_argument('--pooling_strategy', type=str, default='simple_attn',
                    choices=['simple_attn', 'cross_attn', 'gated'])
parser.add_argument('--data_folder', type=str, required=True,
                    help='Path to MR data folder containing subject directories')
parser.add_argument('--jsonl_file', type=str, required=True,
                    help='Path to reports JSONL file')
parser.add_argument('--results_folder', type=str, default='./mr_rate_results')
parser.add_argument('--num_train_steps', type=int, default=100001)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--pretrained_weights', type=str, default=None,
                    help='Path to pretrained RAD-RATE weights to initialize from')
parser.add_argument('--splits_csv', type=str, default=None,
                    help='Path to splits CSV with columns: batch_id, patient_uid, study_uid, split')
parser.add_argument('--split', type=str, default='train',
                    choices=['train', 'val', 'test'],
                    help='Which split to use for training (default: train)')
parser.add_argument('--normalizer', type=str, default='zscore',
                    choices=['zscore', 'percentile', 'minmax'],
                    help='Volume normalization method (default: zscore)')
parser.add_argument('--resume', action='store_true',
                    help='Resume training from latest checkpoint in results_folder')
parser.add_argument('--wandb', action='store_true',
                    help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='mr-rate',
                    help='W&B project name (default: mr-rate)')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='W&B run name (default: auto-generated)')
args = parser.parse_args()

FUSION_MODE = args.fusion_mode
POOLING_STRATEGY = args.pooling_strategy

print(f"--- Configuration ---")
print(f"Fusion Mode: {FUSION_MODE}")
print(f"Pooling Strategy: {POOLING_STRATEGY}")
print(f"Data Folder: {args.data_folder}")
print(f"JSONL File: {args.jsonl_file}")
print(f"Splits CSV: {args.splits_csv}")
print(f"Split: {args.split}")
print(f"Normalizer: {args.normalizer}")
print(f"Resume: {args.resume}")
print(f"W&B: {args.wandb}")

print("\n--- Initializing Tokenizer and Text Encoder ---")
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

print(f"\n--- Initializing VJEPA2 Image Encoder ---")
image_encoder = VJEPA2Encoder(
    input_channels=(3 if FUSION_MODE == "early" else 1),
    freeze_backbone=True,
    use_lora=True,
    lora_r=32,
    lora_alpha=64
)

print(f"\n--- Initializing MR-RATE Model ---")
clip = MRRATE(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    dim_text=768,
    dim_image=image_encoder.output_dim,
    dim_latent=512,
    fusion_mode=FUSION_MODE,
    pooling_strategy=POOLING_STRATEGY,
    use_gradient_checkpointing=True
)

if args.pretrained_weights:
    print(f"\n--- Loading pretrained weights from {args.pretrained_weights} ---")
    clip.load(args.pretrained_weights)

print("\n--- Initializing Trainer ---")
wandb_config = {
    'fusion_mode': FUSION_MODE,
    'pooling_strategy': POOLING_STRATEGY,
    'split': args.split,
    'normalizer': args.normalizer,
    'lr': args.lr,
    'num_train_steps': args.num_train_steps,
}

trainer = MrRateTrainer(
    clip,
    data_folder=args.data_folder,
    jsonl_file=args.jsonl_file,
    splits_csv=args.splits_csv,
    split=args.split,
    batch_size=1,
    num_train_steps=args.num_train_steps,
    lr=args.lr,
    warmup_steps=500,
    save_model_every=500,
    results_folder=args.results_folder,
    normalizer=args.normalizer,
    resume=args.resume,
    use_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name,
    wandb_config=wandb_config,
)

print("\n--- Starting Training ---")
trainer.train()
