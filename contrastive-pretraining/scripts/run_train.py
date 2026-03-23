import argparse
import sys
import os
from transformers import BertTokenizer, BertModel
from mr_rate import MRRATE
from mr_rate_trainer import MrRateTrainer
import torch
import torch.nn as nn


def convert_bn_to_syncbn(module):
    """Convert all BatchNorm layers to SyncBatchNorm for proper distributed training."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(module)


parser = argparse.ArgumentParser(description='MR-RATE Training')
parser.add_argument('--encoder', type=str, default='vjepa2',
                    choices=['vjepa2', 'vjepa21', 'vjepa2_sliding', 'vjepa21_sliding'],
                    help='Vision encoder backbone (default: vjepa2). '
                         'Sliding variants use tiled depth chunks instead of temporal CNN.')
parser.add_argument('--vjepa21_checkpoint', type=str, default=None,
                    help='Path to VJEPA 2.1 pretrained weights (.pt file)')
parser.add_argument('--chunk_size', type=int, default=64,
                    help='Depth chunk size for sliding encoders (default: 64, must be even)')
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
parser.add_argument('--space', type=str, default='native_space',
                    help='Image space subfolder (default: native_space). '
                         'Only used when data is organized as <subject>/<space>/img/')
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
print(f"Encoder: {args.encoder}")
if 'sliding' in args.encoder:
    print(f"Chunk Size: {args.chunk_size}")
print(f"Fusion Mode: {FUSION_MODE}")
print(f"Pooling Strategy: {POOLING_STRATEGY}")
print(f"Data Folder: {args.data_folder}")
print(f"JSONL File: {args.jsonl_file}")
print(f"Splits CSV: {args.splits_csv}")
print(f"Split: {args.split}")
print(f"Space: {args.space}")
print(f"Normalizer: {args.normalizer}")
print(f"Resume: {args.resume}")
print(f"W&B: {args.wandb}")

print("\n--- Initializing Tokenizer and Text Encoder ---")
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

# Setup torch hub for VJEPA 2.1 variants
if 'vjepa21' in args.encoder:
    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, "facebookresearch_vjepa2_main")
    if not os.path.exists(repo_dir):
        print("Downloading VJEPA2 hub repo...")
        torch.hub.list('facebookresearch/vjepa2', force_reload=True)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

if args.encoder == 'vjepa21':
    from vision_encoder import VJEPA21Encoder

    print(f"\n--- Initializing VJEPA 2.1 Image Encoder ---")
    image_encoder = VJEPA21Encoder(
        checkpoint_path=args.vjepa21_checkpoint,
        input_channels=(3 if FUSION_MODE == "early" else 1),
        freeze_backbone=True,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
    )

elif args.encoder == 'vjepa21_sliding':
    from vision_encoder import VJEPA21SlidingEncoder

    print(f"\n--- Initializing VJEPA 2.1 Sliding Encoder (chunk_size={args.chunk_size}) ---")
    image_encoder = VJEPA21SlidingEncoder(
        checkpoint_path=args.vjepa21_checkpoint,
        chunk_size=args.chunk_size,
        input_channels=1,
        freeze_backbone=True,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
    )

elif args.encoder == 'vjepa2_sliding':
    from vision_encoder import VJEPA2SlidingEncoder

    print(f"\n--- Initializing VJEPA2 Sliding Encoder (chunk_size={args.chunk_size}) ---")
    image_encoder = VJEPA2SlidingEncoder(
        chunk_size=args.chunk_size,
        input_channels=1,
        freeze_backbone=True,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
    )

else:  # vjepa2
    from vision_encoder import VJEPA2Encoder

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

# Parameter summary
total_params = sum(p.numel() for p in clip.parameters())
trainable_params = sum(p.numel() for p in clip.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"\n--- Model Parameter Summary ---")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters:    {frozen_params:,}")
print(f"\nPer-module breakdown:")
for name, module in clip.named_children():
    mod_total = sum(p.numel() for p in module.parameters())
    mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name:30s}  total={mod_total:>12,}  trainable={mod_train:>12,}  frozen={mod_total - mod_train:>12,}")

clip = convert_bn_to_syncbn(clip)
print("\nConverted BatchNorm layers to SyncBatchNorm")

if args.pretrained_weights:
    print(f"\n--- Loading pretrained weights from {args.pretrained_weights} ---")
    clip.load(args.pretrained_weights)

print("\n--- Initializing Trainer ---")
wandb_config = {
    'encoder': args.encoder,
    'chunk_size': args.chunk_size if 'sliding' in args.encoder else None,
    'fusion_mode': FUSION_MODE,
    'pooling_strategy': POOLING_STRATEGY,
    'split': args.split,
    'space': args.space,
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
    space=args.space,
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
