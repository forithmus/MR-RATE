"""
MR-RATE Inference Script.

Adapted from RAD-RATE fast_inference_new.py for brain MRI with:
- Variable volumes per subject (2-12+)
- Train/val/test split filtering via splits CSV
- Configurable normalization (zscore, percentile, minmax)
- Text-guided zero-shot classification with configurable pathology prompts
- Optional evaluation if labels are provided

Usage:
    python inference.py \
        --weights_path ./mr_rate_results/MrRate.5000.pt \
        --data_folder /path/to/mri \
        --jsonl_file /path/to/findings_sentences.jsonl \
        --fusion_mode late \
        --splits_csv /path/to/splits.csv \
        --split test \
        --normalizer zscore
"""

import os
import gc
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from einops import rearrange
import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import BertTokenizer, BertModel
from mr_rate import MRRATE
from vision_encoder import VJEPA2Encoder
from data_inference import MRReportDatasetInfer, collate_fn_infer
from eval import evaluate_internal

# ==============================================================================
# OPTIMIZATION SETTINGS
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def load_pathologies(pathologies_file):
    """Load pathologies from JSON file.

    Supports two formats:
      1. {"pathologies": {"Name": {"positive": "...", "negative": "..."}, ...}}
      2. ["Name1", "Name2", ...] (legacy, uses generic prompts)

    Returns list of (label, positive_prompt, negative_prompt) tuples.
    """
    with open(pathologies_file, 'r') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "pathologies" in raw:
        return [
            (name, info["positive"], info["negative"])
            for name, info in raw["pathologies"].items()
        ]
    elif isinstance(raw, list):
        return [
            (p, f"There is {p.lower()}", f"There is no {p.lower()}")
            for p in raw
        ]
    else:
        raise ValueError(
            f"Unsupported pathologies file format. Expected dict with 'pathologies' "
            f"key or a list of strings, got: {type(raw)}"
        )


class MrRateInference(nn.Module):
    """
    MR-RATE inference engine for zero-shot brain MRI classification.

    Given a trained MR-RATE checkpoint, encodes brain MRI volumes and
    computes text-guided similarity scores against pathology prompts.
    """

    def __init__(
        self,
        model: MRRATE,
        *,
        data_folder,
        jsonl_file,
        results_folder='./inference_results',
        fusion_mode="late",
        pooling_strategy="simple_attn",
        normalizer="zscore",
        normalizer_kwargs=None,
        labels_file=None,
        splits_csv=None,
        split="test",
        pathologies,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)

        self.model = model
        self.fusion_mode = fusion_mode
        self.pooling_strategy = pooling_strategy
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.result_folder_txt = str(self.results_folder) + "/"

        # Pathologies: list of (label, positive_prompt, negative_prompt)
        self.pathology_prompts = pathologies
        self.pathologies = [p[0] for p in self.pathology_prompts]

        self.register_buffer('steps', torch.Tensor([0]))

        # Initialize inference dataset
        self.ds = MRReportDatasetInfer(
            data_folder=data_folder,
            jsonl_file=jsonl_file,
            normalizer=normalizer,
            normalizer_kwargs=normalizer_kwargs,
            labels_file=labels_file,
            splits_csv=splits_csv,
            split=split,
        )

        self.device = self.accelerator.device
        self.model.to(self.device)
        self.model.eval()

        # Compile visual transformer if available
        if hasattr(torch, "compile"):
            print("Compiling visual transformer for faster inference...")
            try:
                self.model.visual_transformer = torch.compile(self.model.visual_transformer)
            except Exception as e:
                print(f"Compilation failed (safe to ignore): {e}")

    def _encode_text_latents(self, prompts, device):
        """
        Encode text prompts to latent representations.
        Returns: [N, dim_latent] normalized tensor.
        """
        tokenized = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model.text_seq_len
        ).to(device)

        text_output = self.model.text_transformer(
            tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        )
        enc_text = text_output.last_hidden_state
        text_cls = enc_text[:, 0, :]
        text_latents = self.model.to_text_latent(text_cls)

        return l2norm(text_latents)

    def _encode_visual_tokens(self, images, real_volume_mask, text_latents=None):
        """
        Encode multi-volume brain MRI to visual tokens.

        Args:
            images:            [B, N, 1, D, H, W] where N = variable volumes
            real_volume_mask:  [B, N] boolean
            text_latents:      [B, D] optional text for cross-attn pooling

        Returns:
            visual_tokens: [B, num_tokens, dim_latent] normalized
        """
        b, r, c, d, h, w = images.shape
        vis_proj_layer = self.model.to_visual_latent

        if self.fusion_mode == "early":
            img_in = images.squeeze(2)
            enc = self.model.visual_transformer(img_in)
            visual_tokens = vis_proj_layer(enc)

        elif self.fusion_mode == "mid_cnn":
            flat_img = rearrange(images, 'b r c d h w -> (b r) c d h w')
            cnn_features = self.model.visual_transformer.forward_cnn(flat_img)
            cnn_features = rearrange(cnn_features, '(b r) t h w d -> b r t h w d', r=r)
            m = real_volume_mask.view(b, r, 1, 1, 1, 1).to(cnn_features.dtype)
            merged = (cnn_features * m).sum(1) / m.sum(1).clamp(min=1.0)
            enc = self.model.visual_transformer.forward_transformer(merged)
            visual_tokens = vis_proj_layer(enc)

        elif self.fusion_mode == "late":
            all_tokens_list = []
            for i in range(r):
                enc = self.model.visual_transformer(images[:, i])
                all_tokens_list.append(vis_proj_layer(enc))
            all_tokens = torch.stack(all_tokens_list, dim=1)
            m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
            visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        elif self.fusion_mode == "late_attn":
            all_tokens_list = []
            for i in range(r):
                enc = self.model.visual_transformer(images[:, i])
                all_tokens_list.append(vis_proj_layer(enc))
            all_tokens = torch.stack(all_tokens_list, dim=1)

            if self.pooling_strategy == "simple_attn":
                visual_tokens = self.model.recon_pool(all_tokens, mask=real_volume_mask)
            elif self.pooling_strategy in ["cross_attn", "gated"] and text_latents is not None:
                visual_tokens = self.model.recon_pool(
                    all_tokens, text_latents, mask=real_volume_mask
                )
            else:
                m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
                visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return l2norm(visual_tokens)

    def infer(self, batch_size=1):
        """
        Run zero-shot pathology classification on all subjects.

        For each pathology, computes:
            score = sim("There is {pathology}") - sim("There is no {pathology}")

        Returns dict with predictions, subject_ids, and optionally AUROC results.
        """
        device = self.device
        pathologies = self.pathologies

        # 1. Pre-compute text embeddings: [Pos1, Neg1, Pos2, Neg2, ...]
        print(f"Pre-computing text embeddings for {len(pathologies)} pathologies...")
        prompts = []
        for label, pos_prompt, neg_prompt in self.pathology_prompts:
            prompts.append(pos_prompt)
            prompts.append(neg_prompt)

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                text_latents = self._encode_text_latents(prompts, device=device)
                text_latents = text_latents.to(dtype=torch.bfloat16)
                logit_temp = self.model.logit_temperature.exp().clamp(max=100.0)

        # 2. Inference loop
        print(f"Starting inference (batch_size={batch_size}, "
              f"fusion={self.fusion_mode}, pooling={self.pooling_strategy})...")

        eval_loader = DataLoader(
            self.ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_infer,
            pin_memory=True,
        )

        all_scores = []
        all_labels = []
        all_subject_ids = []
        has_labels = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(eval_loader, desc="Inference")):
                imgs, sentences, subject_id, real_volume_mask, labels = batch

                imgs = imgs.to(device, dtype=torch.bfloat16)
                real_volume_mask = real_volume_mask.to(device)

                with autocast(dtype=torch.bfloat16):
                    # Encode visual tokens
                    if self.pooling_strategy in ["cross_attn", "gated"]:
                        positive_indices = list(range(0, len(prompts), 2))
                        mean_text_query = text_latents[positive_indices].mean(dim=0, keepdim=True)
                        mean_text_query = mean_text_query.expand(imgs.shape[0], -1)
                        visual_tokens = self._encode_visual_tokens(
                            imgs, real_volume_mask, text_latents=mean_text_query
                        )
                    else:
                        visual_tokens = self._encode_visual_tokens(
                            imgs, real_volume_mask
                        )

                    # Text-guided pooling for each pathology query
                    B, N, D = visual_tokens.shape
                    P = text_latents.shape[0]

                    sim_scores = torch.einsum('b n d, p d -> b n p', visual_tokens, text_latents)
                    attn_weights = F.softmax(sim_scores, dim=1)
                    pooled_visual = torch.einsum('b n p, b n d -> b p d', attn_weights, visual_tokens)
                    pooled_visual = l2norm(pooled_visual)

                    final_sim = torch.einsum('b p d, p d -> b p', pooled_visual, text_latents)
                    final_sim = final_sim * logit_temp

                    # Score = logit(Yes) - logit(No)
                    sim_reshaped = final_sim.view(-1, len(pathologies), 2)
                    final_scores = sim_reshaped[:, :, 0] - sim_reshaped[:, :, 1]

                all_scores.append(final_scores.float().cpu().numpy())
                all_subject_ids.append(subject_id)

                if labels.size > 0:
                    has_labels = True
                    all_labels.append(labels)

                n_vols = imgs.shape[1]
                print(f"  [{batch_idx+1}/{len(eval_loader)}] "
                      f"subject={subject_id}, n_vols={n_vols}, "
                      f"scores_range=[{final_scores.min():.2f}, {final_scores.max():.2f}]")

                del imgs, visual_tokens, pooled_visual, final_sim

        # 3. Save results
        final_preds = np.concatenate(all_scores, axis=0)
        plotdir = self.result_folder_txt

        print(f"\nSaving results to {plotdir}...")
        np.savez(f"{plotdir}predicted_scores.npz", data=final_preds)

        with open(f"{plotdir}subject_ids.txt", "w") as f:
            for sid in all_subject_ids:
                f.write(sid + "\n")

        # Save per-subject scores as JSON for easy reading
        results_json = []
        for i, sid in enumerate(all_subject_ids):
            entry = {"subject_id": sid}
            for j, p in enumerate(pathologies):
                entry[p] = float(final_preds[i, j])
            results_json.append(entry)

        with open(f"{plotdir}scores.json", "w") as f:
            json.dump(results_json, f, indent=2)

        # 4. Evaluate if labels available
        aurocs = None
        if has_labels:
            final_labels = np.stack(all_labels, axis=0)
            np.savez(f"{plotdir}labels.npz", data=final_labels)

            print("\nRunning evaluation (AUROC)...")
            aurocs = evaluate_internal(final_preds, final_labels, pathologies, plotdir)

            try:
                import pandas as pd
                aurocs.to_excel(f'{plotdir}aurocs.xlsx', index=False, engine='xlsxwriter')
            except Exception as e:
                print(f"Could not save AUROC Excel (missing xlsxwriter?): {e}")
                aurocs.to_csv(f'{plotdir}aurocs.csv', index=False)

        print("\nInference complete.")
        print(f"  Subjects processed: {len(all_subject_ids)}")
        print(f"  Pathologies scored: {len(pathologies)}")
        print(f"  Results saved to: {plotdir}")

        return {
            'predictions': final_preds,
            'subject_ids': all_subject_ids,
            'labels': np.stack(all_labels, axis=0) if has_labels else None,
            'aurocs': aurocs,
        }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MR-RATE Brain MRI Inference')
    parser.add_argument('--fusion_mode', type=str, required=True,
                        choices=['early', 'mid_cnn', 'late', 'late_attn'])
    parser.add_argument('--pooling_strategy', type=str, default='simple_attn',
                        choices=['simple_attn', 'cross_attn', 'gated'])
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--results_folder', type=str, default='./inference_results')

    # Data paths
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to MR data folder containing subject directories')
    parser.add_argument('--jsonl_file', type=str, required=True,
                        help='Path to reports JSONL file')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to labels CSV (study_uid + binary pathology columns)')

    # Splits and normalization
    parser.add_argument('--splits_csv', type=str, default=None,
                        help='Path to splits CSV with columns: study_uid, split')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--normalizer', type=str, default='zscore',
                        choices=['zscore', 'percentile', 'minmax'])

    # Pathologies definition
    parser.add_argument('--pathologies_file', type=str, required=True,
                        help='JSON file with pathology definitions and positive/negative prompts')

    args = parser.parse_args()

    # Load pathologies
    pathologies = load_pathologies(args.pathologies_file)
    print(f"Loaded {len(pathologies)} pathologies from {args.pathologies_file}")

    print(f"\n{'='*60}")
    print(f"MR-RATE Brain MRI Inference")
    print(f"{'='*60}")
    print(f"Fusion Mode:      {args.fusion_mode}")
    print(f"Pooling Strategy: {args.pooling_strategy}")
    print(f"Splits CSV:       {args.splits_csv}")
    print(f"Split:            {args.split}")
    print(f"Normalizer:       {args.normalizer}")
    print(f"{'='*60}\n")

    # Initialize model
    print("--- Initializing VJEPA2 Image Encoder ---")
    image_encoder = VJEPA2Encoder(
        input_channels=(3 if args.fusion_mode == "early" else 1),
        freeze_backbone=True,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
    )

    print("\n--- Initializing MR-RATE Model ---")
    clip = MRRATE(
        image_encoder=image_encoder,
        dim_image=image_encoder.output_dim,
        dim_text=768,
        dim_latent=512,
        fusion_mode=args.fusion_mode,
        pooling_strategy=args.pooling_strategy,
        use_gradient_checkpointing=False,
    ).cuda()

    print(f"Loading weights: {args.weights_path}")
    clip.load(args.weights_path)

    # Merge LoRA for speed
    print("Merging LoRA weights...")
    try:
        if hasattr(image_encoder, "model") and hasattr(image_encoder.model, "merge_and_unload"):
            image_encoder.model.merge_and_unload()
            print("LoRA merged successfully.")
    except Exception as e:
        print(f"LoRA merge skipped: {e}")

    # Convert to bfloat16
    print("Converting model to bfloat16...")
    clip.to(torch.bfloat16)

    # Run inference
    engine = MrRateInference(
        clip,
        data_folder=args.data_folder,
        jsonl_file=args.jsonl_file,
        results_folder=args.results_folder,
        fusion_mode=args.fusion_mode,
        pooling_strategy=args.pooling_strategy,
        normalizer=args.normalizer,
        labels_file=args.labels_file,
        splits_csv=args.splits_csv,
        split=args.split,
        pathologies=pathologies,
    )

    results = engine.infer(batch_size=args.batch_size)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.results_folder}")
