"""
Extract frozen MR-RATE visual features for linear or MIL probing.

Given a pretrained MR-RATE checkpoint, runs the visual encoder + projection
+ masked pooling over every subject in a split and dumps:

Pooled mode (default) writes the existing linear-probe contract:

  <out_dir>/features_<split>.npy    float32 [N, dim_latent]

Token mode writes a ragged, memory-mappable MIL contract instead:

  <out_dir>/tokens_<split>.bin      concatenated valid projected tokens
  <out_dir>/token_offsets_<split>.npy  int64 [N + 1] bag boundaries
  <out_dir>/token_features_<split>.json cache metadata and provenance

Both modes also write labels, study IDs, and the ordered label schema.

Run this once per split (train / val / test). Then `linear_probe.py`
trains and evaluates a linear classifier on the cached features in
seconds — no need to re-encode the 3D volumes every epoch.

Token mode stops immediately before MR-RATE's final masked mean. It is the
required input for NeuroVFM-style Classify-Then-Aggregate MIL; pooled
`features_*.npy` cannot be used for MIL because the instances are gone.

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
import hashlib
import uuid
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str, include_digest: bool = True) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    record = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_digest:
        record["sha256"] = _sha256_file(resolved)
    return record


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cache_provenance(args, dim_latent: int) -> tuple[dict, str]:
    jsonl = _file_record(args.jsonl_file, include_digest=False)
    provenance = {
        "version": 2,
        "checkpoint": _file_record(args.weights_path),
        "encoder": args.encoder,
        "vjepa21_checkpoint": (
            _file_record(args.vjepa21_checkpoint)
            if args.vjepa21_checkpoint else None
        ),
        "chunk_size": args.chunk_size,
        "fusion_mode": args.fusion_mode,
        "pooling_strategy": args.pooling_strategy,
        "extra_latent_projection": args.extra_latent_projection,
        "dim_latent": dim_latent,
        "normalizer": args.normalizer,
        "space": args.space,
        "use_preprocessed": args.use_preprocessed,
        "preprocessed_dir": (
            str(Path(args.preprocessed_dir).resolve())
            if args.preprocessed_dir else None
        ),
        "data_folder": (
            str(Path(args.data_folder).resolve()) if args.data_folder else None
        ),
        "jsonl": jsonl,
        "labels": _file_record(args.labels_file),
        "splits": _file_record(args.splits_csv),
        "cache_dtype": args.cache_dtype,
        "max_tokens_per_study": args.max_tokens_per_study,
    }
    serialized = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return provenance, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
    import torch as _torch
    from pathlib import Path as _Path

    p = _Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"--weights_path does not exist: {weights_path}")

    # Snapshot a representative set of param hashes pre-load
    def _hashes(model):
        out = {}
        group_counts = {"projection": 0, "pool": 0, "encoder": 0, "text": 0}
        for n, t in model.named_parameters():
            if "to_visual_latent" in n:
                group = "projection"
            elif "recon_pool" in n:
                group = "pool"
            elif "visual_transformer" in n:
                group = "encoder"
            elif "text_transformer.encoder.layer.0" in n:
                group = "text"
            else:
                continue
            if group_counts[group] >= 2:
                continue
            flat = t.detach().reshape(-1)
            if flat.numel() > 4096:
                indices = _torch.linspace(0, flat.numel() - 1, 4096).long()
                flat = flat.index_select(0, indices.to(flat.device))
            value = flat.cpu().float().numpy().tobytes()
            out[n] = hashlib.md5(value).hexdigest()[:8]
            group_counts[group] += 1
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
            "The checkpoint does not match the constructed model parameters."
        )
    projector_prefix = (
        "to_visual_latent_extra."
        if clip.extra_latent_projection else "to_visual_latent."
    )
    critical_prefixes = ("visual_transformer.", projector_prefix)
    critical_missing = [
        name for name in missing if name.startswith(critical_prefixes)
    ]
    if critical_missing:
        raise RuntimeError(
            "Checkpoint is missing MIL-critical visual parameters. "
            f"First entries: {critical_missing[:10]}"
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
        extra_latent_projection=args.extra_latent_projection,
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
    parser.add_argument("--extra_latent_projection", action="store_true",
                        help="Use the checkpoint's extra visual projection for MIL tokens.")
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
    parser.add_argument("--feature_level", type=str, default="pooled",
                        choices=["pooled", "tokens"],
                        help="pooled keeps the linear-probe format; tokens writes "
                             "ragged pre-global-pooling bags for mil_probe.py.")
    parser.add_argument("--cache_dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="On-disk dtype for --feature_level tokens.")
    parser.add_argument("--max_tokens_per_study", type=int, default=0,
                        help="Deterministically subsample each token bag to this size. "
                             "0 preserves every encoder token (exact but potentially very large).")
    parser.add_argument("--strict_missing", action="store_true",
                        help="Abort if the checkpoint is missing any model parameter "
                             "(catches wrong --fusion_mode / --encoder mismatch).")
    args = parser.parse_args()

    if args.use_preprocessed:
        if not args.preprocessed_dir:
            parser.error("--use_preprocessed requires --preprocessed_dir")
    elif not args.data_folder:
        parser.error("--data_folder is required unless --use_preprocessed is set")
    if args.max_tokens_per_study < 0:
        parser.error("--max_tokens_per_study must be >= 0")
    if args.feature_level == "tokens" and args.fusion_mode != "late":
        parser.error(
            "MIL token extraction requires --fusion_mode late so every valid "
            "series can remain a separate set of instances."
        )

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
    if names_path.exists():
        existing_names = json.loads(names_path.read_text())
        if existing_names != ds.label_columns:
            raise RuntimeError(
                f"{names_path} does not match the labels file for this run. "
                "Use a separate --out_dir or remove the stale cache."
            )
    else:
        _atomic_write_text(
            names_path,
            json.dumps(ds.label_columns, indent=2, ensure_ascii=False) + "\n",
        )
        print(f"Wrote {names_path}")

    loader = DataLoader(
        ds, batch_size=1, num_workers=4, shuffle=False,
        drop_last=False, collate_fn=collate_fn_infer, pin_memory=True,
    )

    pooled_feats: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    sids: list[str] = []
    n_unlabeled = 0
    device = next(clip.parameters()).device

    token_dtype = np.dtype(args.cache_dtype)
    token_offsets: list[int] = [0]
    full_token_counts: list[int] = []
    series_counts: list[int] = []
    cache_provenance = cache_fingerprint = None
    token_path = token_tmp_path = None
    token_fh = None
    cache_id = None
    if args.feature_level == "tokens":
        cache_provenance, cache_fingerprint = _cache_provenance(args, dim_latent)
        for other_manifest in out_dir.glob("token_features_*.json"):
            other = json.loads(other_manifest.read_text())
            if other.get("cache_fingerprint") != cache_fingerprint:
                raise RuntimeError(
                    f"Cache configuration differs from {other_manifest}. "
                    "Use a separate --out_dir for a different encoder or preprocessing setup."
                )
        cache_id = uuid.uuid4().hex[:12]
        token_path = out_dir / f"tokens_{args.split}_{cache_id}.bin"
        token_tmp_path = token_path.with_name(token_path.name + f".tmp.{os.getpid()}")
        token_fh = open(token_tmp_path, "wb")

    print(f"\n--- Encoding {len(loader)} subjects ---")
    try:
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc=f"encode[{args.split}]"):
                imgs, _sentences, subject_id, real_volume_mask, labels = batch
                if labels.size == 0:
                    n_unlabeled += 1
                    continue
                imgs = imgs.to(device, dtype=torch.bfloat16)
                real_volume_mask = real_volume_mask.to(device)

                with autocast(dtype=torch.bfloat16):
                    encoded = clip(
                        text_input=None,
                        image=imgs,
                        device=device,
                        real_volume_mask=real_volume_mask,
                        return_loss=False,
                        return_visual_tokens=(args.feature_level == "tokens"),
                    )

                if args.feature_level == "tokens":
                    visual_tokens, token_mask = encoded
                    valid_tokens = visual_tokens[0, token_mask[0]]
                    if valid_tokens.ndim != 2 or valid_tokens.shape[1] != dim_latent:
                        raise RuntimeError(
                            f"Unexpected visual token shape for {subject_id}: "
                            f"{tuple(valid_tokens.shape)}; expected [T, {dim_latent}]"
                        )
                    if valid_tokens.shape[0] == 0:
                        raise RuntimeError(f"Encoder returned an empty token bag for {subject_id}")
                    padded_series = real_volume_mask.shape[1]
                    if visual_tokens.shape[1] % padded_series != 0:
                        raise RuntimeError("Flat MIL tokens cannot be divided into series")
                    tokens_per_series = visual_tokens.shape[1] // padded_series
                    real_series = int(real_volume_mask[0].sum().item())
                    full_token_count = real_series * tokens_per_series
                    if valid_tokens.shape[0] != full_token_count:
                        raise RuntimeError("Series mask and valid token count disagree")
                    full_token_counts.append(full_token_count)
                    series_counts.append(real_series)
                    if (args.max_tokens_per_study > 0 and
                            valid_tokens.shape[0] > args.max_tokens_per_study):
                        keep_array = np.rint(np.linspace(
                            0, valid_tokens.shape[0] - 1,
                            num=args.max_tokens_per_study,
                        )).astype(np.int64)
                        keep = torch.from_numpy(keep_array).to(valid_tokens.device)
                        valid_tokens = valid_tokens.index_select(0, keep)
                    token_array = (
                        valid_tokens.float().cpu().numpy().astype(token_dtype, copy=False)
                    )
                    token_array.tofile(token_fh)
                    token_offsets.append(token_offsets[-1] + token_array.shape[0])
                else:
                    pooled_feats.append(encoded.float().cpu().numpy().reshape(-1))

                labs.append(np.asarray(labels, dtype=np.float32).reshape(-1))
                sids.append(subject_id)

        if not labs:
            raise RuntimeError(f"No labeled subjects encoded for split={args.split}.")
    except BaseException:
        if token_fh is not None:
            token_fh.close()
            token_tmp_path.unlink(missing_ok=True)
        raise

    if token_fh is not None:
        token_fh.flush()
        os.fsync(token_fh.fileno())
        token_fh.close()

    Y = np.stack(labs, axis=0)                  # [N, num_classes]
    assert Y.shape[0] == len(sids)
    assert Y.shape[1] == num_classes, f"label width {Y.shape[1]} != classes {num_classes}"

    print(f"\nWrote:")
    if args.feature_level == "tokens":
        offsets = np.asarray(token_offsets, dtype=np.int64)
        if offsets.shape[0] != Y.shape[0] + 1:
            raise RuntimeError("Token offsets and labels became misaligned")
        os.replace(token_tmp_path, token_path)
        offsets_path = out_dir / f"token_offsets_{args.split}_{cache_id}.npy"
        lab_path = out_dir / f"labels_{args.split}_{cache_id}.npy"
        sid_path = out_dir / f"subject_ids_{args.split}_{cache_id}.txt"
        full_counts_path = out_dir / f"full_token_counts_{args.split}_{cache_id}.npy"
        series_counts_path = out_dir / f"series_counts_{args.split}_{cache_id}.npy"
        metadata_path = out_dir / f"token_features_{args.split}.json"
        _atomic_save_npy(offsets_path, offsets)
        _atomic_save_npy(lab_path, Y)
        _atomic_write_text(sid_path, "\n".join(sids) + "\n")
        _atomic_save_npy(
            full_counts_path, np.asarray(full_token_counts, dtype=np.int64)
        )
        _atomic_save_npy(
            series_counts_path, np.asarray(series_counts, dtype=np.int32)
        )
        metadata = {
            "format": "raw_numpy_memmap",
            "format_version": 2,
            "feature_level": "projected_per_series_visual_tokens",
            "split": args.split,
            "tokens_file": token_path.name,
            "offsets_file": offsets_path.name,
            "labels_file": lab_path.name,
            "subject_ids_file": sid_path.name,
            "full_token_counts_file": full_counts_path.name,
            "series_counts_file": series_counts_path.name,
            "dtype": token_dtype.name,
            "dim": dim_latent,
            "num_studies": len(sids),
            "num_tokens": int(offsets[-1]),
            "max_tokens_per_study": args.max_tokens_per_study,
            "cache_fingerprint": cache_fingerprint,
            "provenance": cache_provenance,
        }
        # Publish the canonical manifest last. Until this atomic replace, any
        # previous cache generation remains internally consistent.
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
        print(f"  {token_path}  shape=({offsets[-1]}, {dim_latent})  dtype={token_dtype.name}")
        print(f"  {offsets_path}  shape={offsets.shape}  dtype={offsets.dtype}")
        print(f"  {metadata_path}")
    else:
        feature_matrix = np.stack(pooled_feats, axis=0)
        assert feature_matrix.shape[0] == Y.shape[0]
        feat_path = out_dir / f"features_{args.split}.npy"
        lab_path = out_dir / f"labels_{args.split}.npy"
        sid_path = out_dir / f"subject_ids_{args.split}.txt"
        _atomic_save_npy(feat_path, feature_matrix)
        _atomic_save_npy(lab_path, Y)
        _atomic_write_text(sid_path, "\n".join(sids) + "\n")
        print(f"  {feat_path}  shape={feature_matrix.shape}  dtype={feature_matrix.dtype}")
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
