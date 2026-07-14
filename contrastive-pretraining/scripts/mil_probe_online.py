#!/usr/bin/env python3
"""Train the exact MIL head while encoding frozen MRI studies online.

This entry point does not write or consume a token-embedding cache. The visual
encoder is reconstructed from the supplied checkpoint, frozen, and run under
torch.no_grad() for every study in every epoch. Only the shared
ClassifyThenAggregate head receives gradients.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_inference import MRReportDatasetInfer, collate_fn_infer
from extract_features import _load_and_verify, build_encoder
from mil_probe import (
    ClassifyThenAggregate,
    auroc_table,
    make_scheduler,
    per_class_metrics,
    select_validation_thresholds,
)


@dataclass
class DatasetMetadata:
    subject_ids: list[str]
    labels: np.ndarray
    label_names: list[str]


@dataclass
class EncodedStudy:
    tokens: torch.Tensor
    target: torch.Tensor
    subject_id: str
    original_token_indices: np.ndarray | None
    full_token_count: int
    series_count: int
    tokens_per_series: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online frozen-encoder Classify-Then-Aggregate MIL"
    )
    model = parser.add_argument_group("frozen encoder")
    model.add_argument("--weights_path", required=True)
    model.add_argument(
        "--encoder",
        default="vjepa2",
        choices=("vjepa2", "vjepa21", "vjepa2_sliding", "vjepa21_sliding"),
    )
    model.add_argument("--vjepa21_checkpoint", default=None)
    model.add_argument("--chunk_size", type=int, default=64)
    model.add_argument("--fusion_mode", default="late", choices=("late",))
    model.add_argument(
        "--pooling_strategy",
        default="simple_attn",
        choices=("simple_attn", "cross_attn", "gated"),
    )
    model.add_argument("--dim_latent", type=int, default=512)
    model.add_argument("--extra_latent_projection", action="store_true")
    model.add_argument("--strict_missing", action="store_true")

    data = parser.add_argument_group("study data")
    data.add_argument("--data_folder", default=None)
    data.add_argument("--jsonl_file", required=True)
    data.add_argument("--labels_file", required=True)
    data.add_argument("--splits_csv", required=True)
    data.add_argument("--space", default="native_space")
    data.add_argument(
        "--normalizer", default="zscore", choices=("zscore", "percentile", "minmax")
    )
    data.add_argument("--preprocessed_dir", default=None)
    data.add_argument("--use_preprocessed", action="store_true")
    data.add_argument("--cache_allow_mismatch", action="store_true")
    data.add_argument(
        "--max_tokens_per_study",
        type=int,
        default=0,
        help="Deterministically subsample after encoding; 0 keeps every token.",
    )

    training = parser.add_argument_group("MIL training")
    training.add_argument("--results_dir", default="./mil_probe_online_results")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--lr", type=float, default=5e-4)
    training.add_argument("--weight_decay", type=float, default=0.05)
    training.add_argument("--warmup_fraction", type=float, default=0.10)
    training.add_argument("--grad_clip", type=float, default=1.0)
    training.add_argument("--grad_accum_steps", type=int, default=4)
    training.add_argument("--hidden_dim", type=int, default=512)
    training.add_argument("--mlp_hidden_dim", type=int, default=384)
    training.add_argument("--no_pos_weight", action="store_true")
    training.add_argument("--num_workers", type=int, default=4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--amp_dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    training.add_argument("--resume", default=None, help="Path to an online last.pt checkpoint.")
    training.add_argument("--save_test_attention", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Online MIL currently requires CUDA because encoder construction is CUDA-only.")
    if args.epochs <= 0 or args.lr <= 0:
        raise ValueError("--epochs and --lr must be positive")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("--weight_decay and --grad_clip cannot be negative")
    if not 0.0 <= args.warmup_fraction <= 1.0:
        raise ValueError("--warmup_fraction must be in [0, 1]")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive")
    if args.hidden_dim <= 0 or args.mlp_hidden_dim <= 0:
        raise ValueError("MIL hidden dimensions must be positive")
    if args.num_workers < 0 or args.max_tokens_per_study < 0:
        raise ValueError("Worker and token counts cannot be negative")
    if args.chunk_size <= 0 or args.dim_latent <= 0:
        raise ValueError("--chunk_size and --dim_latent must be positive")
    if not Path(args.weights_path).is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.weights_path}")
    for name in ("jsonl_file", "labels_file", "splits_csv"):
        value = Path(getattr(args, name))
        if not value.is_file():
            raise FileNotFoundError(f"--{name} not found: {value}")
    if args.use_preprocessed:
        if not args.preprocessed_dir or not Path(args.preprocessed_dir).is_dir():
            raise ValueError("--use_preprocessed requires an existing --preprocessed_dir")
    elif not args.data_folder or not Path(args.data_folder).is_dir():
        raise ValueError("Raw online encoding requires an existing --data_folder")
    if args.vjepa21_checkpoint and not Path(args.vjepa21_checkpoint).is_file():
        raise FileNotFoundError(f"--vjepa21_checkpoint not found: {args.vjepa21_checkpoint}")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_compute_dtype(name: str) -> torch.dtype:
    if name == "auto":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if supports_bf16 else torch.float16
    if name == "bfloat16":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supports_bf16:
            raise RuntimeError("This CUDA device does not support bfloat16; use --amp_dtype float16.")
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def autocast_context(dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.is_file():
        stat = resolved.stat()
        record.update({"size": stat.st_size, "sha256": sha256_file(resolved)})
    return record


def cohort_digest(metadata: DatasetMetadata) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata.subject_ids, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(metadata.labels, dtype=np.float32).tobytes())
    digest.update(json.dumps(metadata.label_names, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dataset(args: argparse.Namespace, split: str) -> MRReportDatasetInfer:
    return MRReportDatasetInfer(
        data_folder=args.data_folder,
        jsonl_file=args.jsonl_file,
        space=args.space,
        normalizer=args.normalizer,
        labels_file=args.labels_file,
        splits_csv=args.splits_csv,
        split=split,
        preprocessed_dir=args.preprocessed_dir,
        use_preprocessed=args.use_preprocessed,
        cache_allow_mismatch=args.cache_allow_mismatch,
    )


def dataset_metadata(dataset: MRReportDatasetInfer, split: str) -> DatasetMetadata:
    if len(dataset) == 0:
        raise ValueError(f"The {split} split contains no eligible studies")
    label_names = [str(name) for name in dataset.label_columns]
    if not label_names:
        raise ValueError("The labels file does not define any output classes")
    subject_ids: list[str] = []
    vectors: list[np.ndarray] = []
    missing: list[str] = []
    for sample in dataset.samples:
        subject_id = str(sample["subject_id"])
        if "labels" not in sample:
            missing.append(subject_id)
            continue
        vector = np.asarray(sample["labels"], dtype=np.float32)
        if vector.shape != (len(label_names),):
            raise ValueError(
                f"{split} study {subject_id!r} has label shape {vector.shape}; "
                f"expected {(len(label_names),)}"
            )
        subject_ids.append(subject_id)
        vectors.append(vector)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"The {split} split has {len(missing)} unlabeled studies: {preview}")
    if len(subject_ids) != len(dataset.samples):
        raise RuntimeError(f"Internal metadata alignment failed for the {split} split")
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(f"Duplicate subject IDs detected within the {split} split")
    labels = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(labels).all():
        raise ValueError(f"The {split} split contains non-finite labels")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError(f"The {split} split contains labels outside strict binary values 0 and 1")
    return DatasetMetadata(subject_ids=subject_ids, labels=labels, label_names=label_names)


def validate_split_pair(
    left: DatasetMetadata,
    left_name: str,
    right: DatasetMetadata,
    right_name: str,
) -> None:
    if left.label_names != right.label_names:
        raise ValueError(f"Label schema mismatch between {left_name} and {right_name}")
    overlap = sorted(set(left.subject_ids).intersection(right.subject_ids))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(
            f"Subject leakage between {left_name} and {right_name}: "
            f"{len(overlap)} overlapping IDs ({preview})"
        )


def make_loader(
    dataset: MRReportDatasetInfer,
    *,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn_infer,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def evenly_spaced_indices(length: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, maximum)).astype(np.int64)


def encode_study(
    encoder: torch.nn.Module,
    batch: tuple[Any, ...],
    device: torch.device,
    compute_dtype: torch.dtype,
    max_tokens_per_study: int,
    *,
    keep_mapping: bool,
) -> EncodedStudy:
    images, _sentences, subject_id, real_volume_mask, labels = batch
    if not isinstance(images, torch.Tensor) or images.ndim != 6 or images.shape[0] != 1:
        raise ValueError("Online MIL expects images with shape [1,R,1,D,H,W]")
    images = images.to(device=device, dtype=compute_dtype, non_blocking=True)
    real_volume_mask = torch.as_tensor(real_volume_mask, dtype=torch.bool, device=device)
    if real_volume_mask.ndim != 2 or real_volume_mask.shape[0] != 1:
        raise ValueError("real_volume_mask must have shape [1,R]")
    with torch.no_grad():
        with autocast_context(compute_dtype):
            visual_tokens, token_mask = encoder(
                text_input=None,
                image=images,
                device=device,
                real_volume_mask=real_volume_mask,
                return_loss=False,
                return_visual_tokens=True,
            )
    if visual_tokens.ndim != 3 or token_mask.shape != visual_tokens.shape[:2]:
        raise RuntimeError("Encoder output must be [B,N,D] plus a matching [B,N] mask")
    valid_tokens = visual_tokens[0, token_mask[0].bool()].detach()
    full_token_count = int(valid_tokens.shape[0])
    if full_token_count <= 0:
        raise ValueError(f"Study {subject_id!r} produced no valid visual tokens")
    series_count = int(real_volume_mask[0].sum().item())
    if series_count <= 0 or full_token_count % series_count != 0:
        raise RuntimeError(
            f"Cannot map {full_token_count} valid tokens across {series_count} valid series"
        )
    tokens_per_series = full_token_count // series_count
    selected_indices = evenly_spaced_indices(full_token_count, max_tokens_per_study)
    if selected_indices.size != full_token_count:
        index_tensor = torch.from_numpy(selected_indices).to(device=valid_tokens.device)
        valid_tokens = valid_tokens.index_select(0, index_tensor)
    if not torch.isfinite(valid_tokens).all():
        raise ValueError(f"Study {subject_id!r} produced non-finite visual tokens")
    if valid_tokens.requires_grad:
        raise RuntimeError("Frozen encoder output unexpectedly requires gradients")
    target = torch.as_tensor(labels, dtype=torch.float32, device=device).reshape(1, -1)
    return EncodedStudy(
        tokens=valid_tokens,
        target=target,
        subject_id=str(subject_id),
        original_token_indices=selected_indices if keep_mapping else None,
        full_token_count=full_token_count,
        series_count=series_count,
        tokens_per_series=tokens_per_series,
    )


def cumulative_lengths(token_count: int) -> torch.Tensor:
    return torch.tensor([0, token_count], dtype=torch.long)


def verify_gradient_boundary(encoder: torch.nn.Module, head: torch.nn.Module) -> None:
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen encoder accumulated gradients")
    if not any(parameter.grad is not None for parameter in head.parameters()):
        raise RuntimeError("MIL head received no gradients")


def train_one_epoch(
    encoder: torch.nn.Module,
    head: ClassifyThenAggregate,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    compute_dtype: torch.dtype,
    grad_accum_steps: int,
    grad_clip: float,
    max_tokens_per_study: int,
    global_step: int,
    gradient_boundary_checked: bool,
) -> tuple[float, int, bool]:
    encoder.eval()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_studies = len(loader)
    for batch_index, batch in enumerate(loader):
        study = encode_study(
            encoder, batch, device, compute_dtype, max_tokens_per_study, keep_mapping=False
        )
        if study.target.shape[1] != head.n_classes:
            raise ValueError(
                f"Study {study.subject_id!r} has {study.target.shape[1]} labels; "
                f"the MIL head expects {head.n_classes}"
            )
        block_start = (batch_index // grad_accum_steps) * grad_accum_steps
        block_size = min(grad_accum_steps, total_studies - block_start)
        with autocast_context(compute_dtype):
            logits = head(study.tokens, cumulative_lengths(study.tokens.shape[0]))
        loss = criterion(logits.float(), study.target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss for study {study.subject_id!r}")
        total_loss += float(loss.detach().cpu())
        scaler.scale(loss / block_size).backward()
        if not gradient_boundary_checked:
            verify_gradient_boundary(encoder, head)
            gradient_boundary_checked = True
        at_boundary = (batch_index + 1) % grad_accum_steps == 0
        at_end = batch_index + 1 == total_studies
        if at_boundary or at_end:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
    return total_loss / total_studies, global_step, gradient_boundary_checked


def safe_subject_id(subject_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject_id).strip("._")
    return cleaned or "study"


def atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def evaluate_online(
    encoder: torch.nn.Module,
    head: ClassifyThenAggregate,
    loader: DataLoader,
    device: torch.device,
    compute_dtype: torch.dtype,
    max_tokens_per_study: int,
    attention_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    encoder.eval()
    head.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    subject_ids: list[str] = []
    if attention_dir is not None:
        if attention_dir.exists() and any(attention_dir.iterdir()):
            raise FileExistsError(f"Refusing to mix attention exports in {attention_dir}")
        attention_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for dataset_index, batch in enumerate(loader):
            study = encode_study(
                encoder,
                batch,
                device,
                compute_dtype,
                max_tokens_per_study,
                keep_mapping=attention_dir is not None,
            )
            cu_seqlens = cumulative_lengths(study.tokens.shape[0])
            with autocast_context(compute_dtype):
                if attention_dir is None:
                    logits = head(study.tokens, cu_seqlens)
                else:
                    logits, attention, patch_logits = head(
                        study.tokens, cu_seqlens, return_details=True
                    )
            logits_out.append(logits.float().cpu().numpy())
            labels_out.append(study.target.float().cpu().numpy())
            subject_ids.append(study.subject_id)
            if attention_dir is not None:
                if study.original_token_indices is None:
                    raise RuntimeError("Attention export is missing token-index metadata")
                output_path = attention_dir / (
                    f"{dataset_index:06d}_{safe_subject_id(study.subject_id)}.npz"
                )
                atomic_savez(
                    output_path,
                    attention=attention.float().cpu().numpy().astype(np.float16),
                    patch_logits=patch_logits.float().cpu().numpy().astype(np.float16),
                    original_token_indices=study.original_token_indices,
                    full_token_count=np.asarray(study.full_token_count, dtype=np.int64),
                    series_count=np.asarray(study.series_count, dtype=np.int32),
                    tokens_per_series=np.asarray(study.tokens_per_series, dtype=np.int32),
                )
    if not logits_out:
        raise ValueError("Cannot evaluate an empty study loader")
    return np.concatenate(logits_out), np.concatenate(labels_out), subject_ids


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "train_generator": train_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["train_generator"])


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def make_provenance(
    args: argparse.Namespace,
    train_metadata: DatasetMetadata,
    val_metadata: DatasetMetadata,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "online_frozen_encoder_mil",
        "encoder_checkpoint": file_record(args.weights_path),
        "vjepa21_checkpoint": file_record(args.vjepa21_checkpoint),
        "encoder": {
            "name": args.encoder,
            "chunk_size": args.chunk_size,
            "fusion_mode": args.fusion_mode,
            "pooling_strategy": args.pooling_strategy,
            "dim_latent": args.dim_latent,
            "extra_latent_projection": args.extra_latent_projection,
            "strict_missing": args.strict_missing,
        },
        "data": {
            "data_folder": str(Path(args.data_folder).resolve()) if args.data_folder else None,
            "jsonl_file": file_record(args.jsonl_file),
            "labels_file": file_record(args.labels_file),
            "splits_csv": file_record(args.splits_csv),
            "space": args.space,
            "normalizer": args.normalizer,
            "use_preprocessed": args.use_preprocessed,
            "preprocessed_dir": (
                str(Path(args.preprocessed_dir).resolve()) if args.preprocessed_dir else None
            ),
            "cache_allow_mismatch": args.cache_allow_mismatch,
        },
        "cohorts": {
            "train": {"count": len(train_metadata.subject_ids), "digest": cohort_digest(train_metadata)},
            "val": {"count": len(val_metadata.subject_ids), "digest": cohort_digest(val_metadata)},
        },
        "label_names": train_metadata.label_names,
    }


def immutable_description(
    args: argparse.Namespace,
    provenance: dict[str, Any],
    dim_latent: int,
    n_classes: int,
    optimizer_steps_per_epoch: int,
    resolved_dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "provenance": provenance,
        "architecture": {
            "dim": dim_latent,
            "n_classes": n_classes,
            "hidden_dim": args.hidden_dim,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "drop_rate": 0.0,
            "use_gating": True,
            "use_norm": False,
            "use_output_bias_scale": True,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_fraction": args.warmup_fraction,
            "grad_clip": args.grad_clip,
            "grad_accum_steps": args.grad_accum_steps,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "no_pos_weight": args.no_pos_weight,
            "seed": args.seed,
            "amp_dtype": args.amp_dtype,
            "resolved_dtype": str(resolved_dtype),
            "max_tokens_per_study": args.max_tokens_per_study,
        },
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metrics CSV")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    results_dir = Path(args.results_dir)
    if results_dir.exists() and any(results_dir.iterdir()) and args.resume is None:
        raise FileExistsError(
            f"Refusing to overwrite nonempty results directory: {results_dir}. "
            "Use a new directory or --resume."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    compute_dtype = resolve_compute_dtype(args.amp_dtype)

    train_dataset = build_dataset(args, "train")
    val_dataset = build_dataset(args, "val")
    train_metadata = dataset_metadata(train_dataset, "train")
    val_metadata = dataset_metadata(val_dataset, "val")
    validate_split_pair(train_metadata, "train", val_metadata, "val")
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        train_dataset, shuffle=True, num_workers=args.num_workers, generator=train_generator
    )
    val_loader = make_loader(val_dataset, shuffle=False, num_workers=args.num_workers)

    print("Building frozen visual encoder")
    encoder, dim_latent = build_encoder(args)
    _load_and_verify(encoder, args.weights_path, strict_missing=args.strict_missing)
    try:
        image_encoder = encoder.visual_transformer
        if hasattr(image_encoder, "model") and hasattr(image_encoder.model, "merge_and_unload"):
            image_encoder.model.merge_and_unload()
    except Exception as error:
        print(f"LoRA merge skipped: {error}")
    encoder.requires_grad_(False)
    encoder.to(device=device, dtype=compute_dtype)
    encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Failed to freeze every visual-encoder parameter")

    n_classes = len(train_metadata.label_names)
    head = ClassifyThenAggregate(
        dim=dim_latent,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        mlp_hidden_dims=(args.mlp_hidden_dim,),
        drop_rate=0.0,
        init_std=0.02,
        use_gating=True,
        use_norm=False,
        use_output_bias_scale=True,
    ).to(device)
    if args.no_pos_weight:
        pos_weight = None
    else:
        positives = train_metadata.labels.sum(axis=0)
        negatives = len(train_metadata.labels) - positives
        weights = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 100.0)
        pos_weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = make_scheduler(
        optimizer, optimizer_steps_per_epoch * args.epochs, args.warmup_fraction
    )
    scaler = make_grad_scaler(compute_dtype == torch.float16)
    provenance = make_provenance(args, train_metadata, val_metadata)
    immutable = immutable_description(
        args,
        provenance,
        dim_latent,
        n_classes,
        optimizer_steps_per_epoch,
        compute_dtype,
    )
    run_fingerprint = fingerprint(immutable)

    start_epoch = 0
    global_step = 0
    best_epoch = -1
    best_val_mean_auroc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    gradient_boundary_checked = False
    if args.resume is not None:
        resume = load_torch_checkpoint(args.resume, device)
        if resume.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "Resume checkpoint does not match the encoder, data, head, or training configuration"
            )
        head.load_state_dict(resume["head_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["completed_epoch"]) + 1
        global_step = int(resume["global_step"])
        best_epoch = int(resume["best_epoch"])
        best_val_mean_auroc = float(resume["best_val_mean_auroc"])
        best_state = resume.get("best_state_dict")
        history = list(resume["history"])
        gradient_boundary_checked = bool(resume.get("gradient_boundary_checked", False))
        restore_rng_state(resume["rng_state"], train_generator)
        if start_epoch > args.epochs:
            raise ValueError("Resume checkpoint is beyond the configured number of epochs")
        print(f"Resuming at epoch {start_epoch + 1} of {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step, gradient_boundary_checked = train_one_epoch(
            encoder,
            head,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            compute_dtype,
            args.grad_accum_steps,
            args.grad_clip,
            args.max_tokens_per_study,
            global_step,
            gradient_boundary_checked,
        )
        val_logits, val_labels, _ = evaluate_online(
            encoder,
            head,
            val_loader,
            device,
            compute_dtype,
            args.max_tokens_per_study,
        )
        val_mean_auroc, val_per_class = auroc_table(
            val_logits, val_labels, train_metadata.label_names
        )
        history.append(
            json_safe(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_mean_auroc": val_mean_auroc,
                    "val_per_class_auroc": val_per_class,
                    "optimizer_steps": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        )
        if math.isfinite(val_mean_auroc) and val_mean_auroc > best_val_mean_auroc:
            best_val_mean_auroc = float(val_mean_auroc)
            best_epoch = epoch + 1
            best_state = cpu_state_dict(head)
            atomic_torch_save(
                {
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_val_mean_auroc": best_val_mean_auroc,
                    "label_names": train_metadata.label_names,
                    "run_fingerprint": run_fingerprint,
                    "provenance": provenance,
                },
                results_dir / "best.pt",
            )
        atomic_torch_save(
            {
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_mean_auroc": best_val_mean_auroc,
                "best_state_dict": best_state,
                "history": history,
                "gradient_boundary_checked": gradient_boundary_checked,
                "rng_state": rng_state(train_generator),
                "run_fingerprint": run_fingerprint,
                "immutable_run": immutable,
                "args": vars(args),
            },
            results_dir / "last.pt",
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.6f} val_mean_auroc={val_mean_auroc:.6f}"
        )

    if best_state is None or best_epoch < 1:
        raise RuntimeError(
            "Validation AUROC was non-finite in every epoch; no test evaluation was performed"
        )
    head.load_state_dict(best_state)
    head.to(device)
    best_val_logits, best_val_labels, _ = evaluate_online(
        encoder,
        head,
        val_loader,
        device,
        compute_dtype,
        args.max_tokens_per_study,
    )
    validation_thresholds = select_validation_thresholds(best_val_logits, best_val_labels)
    if not np.isfinite(validation_thresholds).all():
        raise RuntimeError("Validation threshold selection produced non-finite values")

    test_dataset = build_dataset(args, "test")
    test_metadata = dataset_metadata(test_dataset, "test")
    validate_split_pair(train_metadata, "train", test_metadata, "test")
    validate_split_pair(val_metadata, "val", test_metadata, "test")
    test_loader = make_loader(test_dataset, shuffle=False, num_workers=args.num_workers)
    attention_dir = results_dir / "test_attention" if args.save_test_attention else None
    test_logits, test_labels, test_subject_ids = evaluate_online(
        encoder,
        head,
        test_loader,
        device,
        compute_dtype,
        args.max_tokens_per_study,
        attention_dir=attention_dir,
    )
    if test_subject_ids != test_metadata.subject_ids:
        raise RuntimeError("Test DataLoader order does not match validated test metadata")
    test_mean_auroc, test_per_class = auroc_table(
        test_logits, test_labels, train_metadata.label_names
    )
    test_metric_rows = per_class_metrics(
        test_logits, test_labels, train_metadata.label_names, validation_thresholds
    )

    final_provenance = copy.deepcopy(provenance)
    final_provenance["cohorts"]["test"] = {
        "count": len(test_metadata.subject_ids),
        "digest": cohort_digest(test_metadata),
    }
    architecture = {
        "name": "ClassifyThenAggregate",
        "dim": dim_latent,
        "n_classes": n_classes,
        "hidden_dim": args.hidden_dim,
        "mlp_hidden_dims": [args.mlp_hidden_dim],
        "drop_rate": 0.0,
        "init_std": 0.02,
        "use_gating": True,
        "use_norm": False,
        "use_output_bias_scale": True,
    }
    atomic_torch_save(
        {
            "state_dict": cpu_state_dict(head),
            "architecture": architecture,
            "label_names": train_metadata.label_names,
            "data_provenance": final_provenance,
            "best_epoch": best_epoch,
            "best_val_mean_auroc": best_val_mean_auroc,
            "validation_thresholds": validation_thresholds.tolist(),
            "args": vars(args),
        },
        results_dir / "mil_head.pt",
    )
    atomic_save_npy(results_dir / "test_logits.npy", test_logits.astype(np.float32))
    atomic_save_npy(results_dir / "test_labels.npy", test_labels.astype(np.float32))
    atomic_write_text(results_dir / "test_subject_ids.txt", "\n".join(test_subject_ids) + "\n")
    atomic_write_text(
        results_dir / "history.json",
        json.dumps(json_safe(history), indent=2, allow_nan=False) + "\n",
    )
    threshold_payload = dict(zip(train_metadata.label_names, validation_thresholds.tolist()))
    atomic_write_text(
        results_dir / "validation_thresholds.json",
        json.dumps(json_safe(threshold_payload), indent=2, allow_nan=False) + "\n",
    )
    metric_payload = {
        "mean_auroc": test_mean_auroc,
        "per_class": test_per_class,
        "metrics": test_metric_rows,
    }
    atomic_write_text(
        results_dir / "per_class_test_auroc.json",
        json.dumps(json_safe(metric_payload), indent=2, allow_nan=False) + "\n",
    )
    save_csv(results_dir / "test_aurocs.csv", test_metric_rows)
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation mean AUROC: {best_val_mean_auroc:.6f}")
    print(f"Test mean AUROC: {test_mean_auroc:.6f}")
    print(f"Results written to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
