"""
Extract frozen MR-RATE visual features for linear probing.

Given a pretrained MR-RATE checkpoint, runs the visual encoder + projection
+ masked pooling over every subject in a split and dumps:

  <out_dir>/features_<split>.npy   float32 [N, dim_latent]
  <out_dir>/labels_<split>.npy     float32 [N, num_classes]
  <out_dir>/subject_ids_<split>.txt one study_uid per line
  <out_dir>/label_names.json       list of 32 pathology names
                                   (only written on the first run)

Run this once per split (train / val / test). Then `linear_probe.py`
trains and evaluates a linear classifier on the cached features in
seconds — no need to re-encode the 3D volumes every epoch.

The frozen backbone returns `l2norm(masked_mean(visual_tokens))`, the
same global representation that `inference.py` uses for zero-shot scoring.

Usage:
    python extract_features.py \
        --weights_path ./mr_rate_results/MrRate.5000.pt \
        --data_folder /path/to/mri \
        --jsonl_file /path/to/findings_sentences.jsonl \
        --labels_file .../splits_agreement/mrrate_labels.csv \
        --splits_csv  .../splits_agreement/splits.csv \
        --split test \
        --fusion_mode late \
        --out_dir ./linear_probe_features
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import tqdm

from mr_rate import MRRATE
from data_inference import MRReportDatasetInfer, collate_fn_infer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _load_and_verify(clip: "MRRATE", weights_path: str, strict_missing: bool = False) -> None:
    """Load checkpoint into MRRATE and report exactly what was matched.

    `MRRATE.load` uses `strict=False` and silently skips mismatched keys.
    To catch a wrong-checkpoint or wrong-fusion-mode situation early, this
    helper:
      1) Compares pre/post-load weight hashes for a sampled set of encoder
         params and aborts if NOTHING actually changed (= silent no-op).
      2) Prints missing / unexpected key counts (and the first few names).
      3) Optionally aborts on any missing key (--strict_missing) so a typo
         in --fusion_mode (which changes module names) fails loudly.
    """
    import hashlib
    import torch as _torch
    from pathlib import Path as _Path

    p = _Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"--weights_path does not exist: {weights_path}")

    # Snapshot a representative set of param hashes pre-load
    def _hashes(model):
        out = {}
        for n, t in model.named_parameters():
            # Sample only a handful for speed; covers projections, pooling, encoder
            if any(s in n for s in ("to_visual_latent", "to_text_latent",
                                    "recon_pool", "visual_transformer",
                                    "text_transformer.encoder.layer.0")):
                out[n] = hashlib.md5(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:8]
        return out

    pre = _hashes(clip)

    # Load via MRRATE.load (handles 'module.' prefix stripping), then also
    # load_state_dict ourselves to capture the missing/unexpected key report.
    clip.load(str(p))
    pt = _torch.load(str(p), map_location="cpu")
    clean = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in pt.items()}
    incompat = clip.load_state_dict(clean, strict=False)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)

    post = _hashes(clip)
    changed = sum(1 for n in pre if pre[n] != post.get(n))
    total = len(pre)

    print(f"[load] checkpoint: {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
    print(f"[load] sampled params changed by load: {changed}/{total}")
    print(f"[load] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    if missing:
        head = ", ".join(missing[:5]) + (f", ... (+{len(missing)-5} more)" if len(missing) > 5 else "")
        print(f"[load]   first missing: {head}")
    if unexpected:
        head = ", ".join(unexpected[:5]) + (f", ... (+{len(unexpected)-5} more)" if len(unexpected) > 5 else "")
        print(f"[load]   first unexpected: {head}")

    if changed == 0:
        raise RuntimeError(
            f"No model parameters changed when loading {weights_path}. "
            "The checkpoint likely doesn't match the architecture "
            "(wrong --fusion_mode / --encoder / --dim_latent)."
        )
    if strict_missing and missing:
        raise RuntimeError(
            f"--strict_missing set: {len(missing)} keys not present in checkpoint. "
            f"First few: {missing[:5]}"
        )


def build_encoder(args) -> tuple[MRRATE, int]:
    """Mirror run_train.py's encoder selection so the checkpoint loads cleanly."""
    if "vjepa21" in args.encoder:
        import sys
        hub_dir = torch.hub.get_dir()
        repo_dir = os.path.join(hub_dir, "facebookresearch_vjepa2_main")
        if not os.path.exists(repo_dir):
            torch.hub.list("facebookresearch/vjepa2", force_reload=True)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

    if args.encoder == "vjepa21":
        from vision_encoder import VJEPA21Encoder
        image_encoder = VJEPA21Encoder(
            checkpoint_path=args.vjepa21_checkpoint,
            input_channels=(3 if args.fusion_mode == "early" else 1),
            freeze_backbone=True, use_lora=True,
            lora_r=32, lora_alpha=64, lora_dropout=0.05,
        )
    elif args.encoder == "vjepa21_sliding":
        from vision_encoder import VJEPA21SlidingEncoder
        image_encoder = VJEPA21SlidingEncoder(
            checkpoint_path=args.vjepa21_checkpoint,
            chunk_size=args.chunk_size, input_channels=1,
            freeze_backbone=True, use_lora=True,
            lora_r=32, lora_alpha=64, lora_dropout=0.05,
        )
    elif args.encoder == "vjepa2_sliding":
        from vision_encoder import VJEPA2SlidingEncoder
        image_encoder = VJEPA2SlidingEncoder(
            chunk_size=args.chunk_size, input_channels=1,
            freeze_backbone=True, use_lora=True,
            lora_r=32, lora_alpha=64,
        )
    else:
        from vision_encoder import VJEPA2Encoder
        image_encoder = VJEPA2Encoder(
            input_channels=(3 if args.fusion_mode == "early" else 1),
            freeze_backbone=True, use_lora=True, lora_r=32, lora_alpha=64,
        )

    clip = MRRATE(
        image_encoder=image_encoder,
        dim_image=image_encoder.output_dim,
        dim_text=768,
        dim_latent=args.dim_latent,
        fusion_mode=args.fusion_mode,
        pooling_strategy=args.pooling_strategy,
        use_gradient_checkpointing=False,
    ).cuda()
    return clip, args.dim_latent


def main() -> None:
    parser = argparse.ArgumentParser("MR-RATE: extract frozen features for linear probing")
    # Model
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="vjepa2",
                        choices=["vjepa2", "vjepa21", "vjepa2_sliding", "vjepa21_sliding"])
    parser.add_argument("--vjepa21_checkpoint", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--fusion_mode", type=str, required=True,
                        choices=["early", "mid_cnn", "late", "late_attn"])
    parser.add_argument("--pooling_strategy", type=str, default="simple_attn",
                        choices=["simple_attn", "cross_attn", "gated"])
    parser.add_argument("--dim_latent", type=int, default=512)
    # Data
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Raw MR data folder. Required unless --use_preprocessed.")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--labels_file", type=str, required=True,
                        help="study_uid + per-class binary columns (e.g. mrrate_labels.csv)")
    parser.add_argument("--splits_csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--space", type=str, default="native_space")
    parser.add_argument("--normalizer", type=str, default="zscore",
                        choices=["zscore", "percentile", "minmax"])
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                        help="Root of precomputed .npz volumes (preprocess_volumes.py).")
    parser.add_argument("--use_preprocessed", action="store_true",
                        help="Read preprocessed .npz instead of raw NIfTI.")
    parser.add_argument("--cache_allow_mismatch", action="store_true",
                        help="Downgrade a cache-manifest config mismatch to a warning.")
    # Output
    parser.add_argument("--out_dir", type=str, default="./linear_probe_features")
    parser.add_argument("--strict_missing", action="store_true",
                        help="Abort if the checkpoint is missing any model parameter "
                             "(catches wrong --fusion_mode / --encoder mismatch).")
    args = parser.parse_args()

    if args.use_preprocessed:
        if not args.preprocessed_dir:
            parser.error("--use_preprocessed requires --preprocessed_dir")
    elif not args.data_folder:
        parser.error("--data_folder is required unless --use_preprocessed is set")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Building model ({args.encoder}, fusion={args.fusion_mode}) ---")
    clip, dim_latent = build_encoder(args)
    print(f"Loading weights from {args.weights_path}")
    _load_and_verify(clip, args.weights_path, strict_missing=args.strict_missing)

    # Merge LoRA for speed if available
    try:
        ie = clip.visual_transformer
        if hasattr(ie, "model") and hasattr(ie.model, "merge_and_unload"):
            ie.model.merge_and_unload()
            print("LoRA merged.")
    except Exception as e:
        print(f"LoRA merge skipped: {e}")

    clip.to(torch.bfloat16)
    clip.eval()

    print(f"\n--- Dataset (split={args.split}) ---")
    ds = MRReportDatasetInfer(
        data_folder=args.data_folder,
        jsonl_file=args.jsonl_file,
        space=args.space,
        normalizer=args.normalizer,
        labels_file=args.labels_file,
        splits_csv=args.splits_csv,
        split=args.split,
        preprocessed_dir=args.preprocessed_dir,
        use_preprocessed=args.use_preprocessed,
        cache_allow_mismatch=args.cache_allow_mismatch,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No subjects found for split={args.split}.")
    if not ds.label_columns:
        raise RuntimeError("Labels CSV produced 0 columns — check --labels_file.")
    num_classes = len(ds.label_columns)
    print(f"Subjects: {len(ds)}  |  classes: {num_classes}")

    # Persist label names so the linear-probe trainer doesn't need the source CSV
    names_path = out_dir / "label_names.json"
    if not names_path.exists():
        names_path.write_text(json.dumps(ds.label_columns, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {names_path}")

    loader = DataLoader(
        ds, batch_size=1, num_workers=4, shuffle=False,
        drop_last=False, collate_fn=collate_fn_infer, pin_memory=True,
    )

    feats: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    sids: list[str] = []
    n_unlabeled = 0
    device = next(clip.parameters()).device

    print(f"\n--- Encoding {len(loader)} subjects ---")
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=f"encode[{args.split}]"):
            imgs, _sentences, subject_id, real_volume_mask, labels = batch
            if labels.size == 0:
                n_unlabeled += 1
                continue
            imgs = imgs.to(device, dtype=torch.bfloat16)
            real_volume_mask = real_volume_mask.to(device)

            with autocast(dtype=torch.bfloat16):
                # MRRATE in inference mode returns the masked-mean pooled,
                # L2-normalized latent: [1, dim_latent]
                pooled = clip(
                    text_input=None,
                    image=imgs,
                    device=device,
                    real_volume_mask=real_volume_mask,
                    return_loss=False,
                )

            feats.append(pooled.float().cpu().numpy().reshape(-1))
            labs.append(np.asarray(labels, dtype=np.float32).reshape(-1))
            sids.append(subject_id)

    if not feats:
        raise RuntimeError(f"No labeled subjects encoded for split={args.split}.")

    F = np.stack(feats, axis=0)                 # [N, dim_latent]
    Y = np.stack(labs, axis=0)                  # [N, num_classes]
    assert F.shape[0] == Y.shape[0] == len(sids)
    assert Y.shape[1] == num_classes, f"label width {Y.shape[1]} != classes {num_classes}"

    feat_path = out_dir / f"features_{args.split}.npy"
    lab_path = out_dir / f"labels_{args.split}.npy"
    sid_path = out_dir / f"subject_ids_{args.split}.txt"
    np.save(feat_path, F)
    np.save(lab_path, Y)
    sid_path.write_text("\n".join(sids) + "\n")

    print(f"\nWrote:")
    print(f"  {feat_path}  shape={F.shape}  dtype={F.dtype}")
    print(f"  {lab_path}   shape={Y.shape}  dtype={Y.dtype}")
    print(f"  {sid_path}   ({len(sids)} ids)")
    if n_unlabeled:
        print(f"Skipped {n_unlabeled} subjects with no labels in {args.labels_file}.")
    print(f"\nPositives per class (top 10):")
    pos = Y.sum(0).astype(int)
    order = np.argsort(-pos)
    for j in order[:10]:
        print(f"  {ds.label_columns[j]:50s}  {pos[j]:5d}  ({pos[j]/len(Y)*100:.2f}%)")


if __name__ == "__main__":
    main()
