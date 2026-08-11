"""Generate findings reports for validation/test studies from a trained writer.

The generation prefix (visual tokens -> resampler -> projection, MIL
conditioning tokens, report prompt) is identical to training; decoding is
greedy. Output is a CSV with one row per study:

    study_uid, findings_gt, findings_pred

Cached mode reads the exact ragged token cache built by
``mrrate_report_training.build_cache --split val|test``. Online mode runs the
frozen upstream encoder on the fly. Large runs can be sharded across
independent processes with --num-shards/--shard-index; each shard writes its
own CSV, and the evaluator accepts multiple CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .cache import ExactRaggedTokenDataset
from .config import load_config
from .mil import infer_mil, load_frozen_mil
from .model import (
    ReportWriter,
    build_gemma_writer,
    label_semantic_embeddings,
    trainable_state_dict,
)
from .provenance import verify_mil_encoder_provenance
from .targets import load_target_index


def load_writer_checkpoint(
    path: str | Path, model: torch.nn.Module, label_names: list[str]
) -> dict:
    """Load trainable inference weights and fail loudly on schema drift."""

    package = torch.load(path, map_location="cpu", weights_only=False)
    stored_mode = (
        package.get("config", {}).get("writer", {}).get("mil_conditioning")
        or "all_classes"
    )
    model_mode = getattr(model, "mil_conditioning", "all_classes")
    if stored_mode != model_mode:
        raise ValueError(
            f"Checkpoint was trained with mil_conditioning={stored_mode!r} "
            f"but the model is configured for {model_mode!r}"
        )
    if package.get("label_names") != list(label_names):
        raise ValueError("Checkpoint MIL label schema differs from the MIL head")
    state = package.get("trainable_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: missing trainable_state_dict")
    expected = set(trainable_state_dict(model))
    provided = set(state)
    if provided != expected:
        missing = sorted(expected - provided)[:5]
        unexpected = sorted(provided - expected)[:5]
        raise ValueError(
            f"Checkpoint trainable tensors differ from the model: "
            f"missing={missing} unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected checkpoint tensors: {list(incompatible.unexpected_keys)[:5]}"
        )
    return package


def shard_indices(length: int, num_shards: int, shard_index: int) -> list[int]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard arguments")
    return list(range(shard_index, length, num_shards))


def shard_output_path(path: Path, num_shards: int, shard_index: int) -> Path:
    """Every shard writes its own CSV so concurrent jobs never collide."""

    if num_shards <= 1:
        return path
    return path.with_name(
        f"{path.stem}.shard{shard_index:02d}of{num_shards:02d}{path.suffix}"
    )


def completed_study_uids(path: Path) -> set[str]:
    if not path.exists() or not path.stat().st_size:
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["study_uid", "findings_gt", "findings_pred"]:
            raise ValueError(f"{path}: unexpected columns; cannot resume")
        return {str(row["study_uid"]) for row in reader}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("online", "cached"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-studies", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output CSV, skipping completed studies",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output CSV",
    )
    parser.add_argument("--llm-path")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--mil-checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.llm_path:
        config["llm_path"] = args.llm_path
    if args.encoder_checkpoint:
        config["encoder_checkpoint"] = args.encoder_checkpoint
    if args.mil_checkpoint:
        config["mil_checkpoint"] = args.mil_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("MR-RATE report generation requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    targets = load_target_index(config["data"]["reports_csv"])
    mil_mode = str(config["writer"].get("mil_conditioning", "all_classes"))
    if mil_mode == "none":
        # No-classification-labels ablation: visual conditioning only.
        mil_head, label_names, thresholds = None, [], None
    else:
        mil_head, label_names, thresholds = load_frozen_mil(
            config["mil_checkpoint"],
            config["upstream_root"],
            expected_dim=int(config["encoder"]["dim_latent"]),
        )
        mil_head.to(device).eval()
        thresholds = thresholds.to(device)

    if args.mode == "cached":
        source = ExactRaggedTokenDataset(
            config["data"]["cached_tokens_dir"],
            args.split,
            targets,
            expected_dim=int(config["encoder"]["dim_latent"]),
            expected_label_names=(
                label_names if mil_mode == "all_classes" else None
            ),
        )
        subject_ids = source.subject_ids
    else:
        from .online import OnlineSource

        source = OnlineSource(config, device, split=args.split)
        subject_ids = source.subject_ids
        missing = [value for value in subject_ids if value not in targets]
        if missing:
            raise ValueError(
                f"{len(missing)} online studies lack report targets; "
                f"first={missing[:5]}"
            )
    if mil_mode == "none":
        provenance = {"skipped": "mil_conditioning=none has no MIL provenance"}
    else:
        provenance = verify_mil_encoder_provenance(
            config["mil_checkpoint"],
            config["encoder_checkpoint"],
            config["encoder"],
            cache_metadata=source.metadata if args.mode == "cached" else None,
        )
    print("[provenance] " + json.dumps(provenance), flush=True)

    writer_config = config["writer"]
    llm, tokenizer, hidden_size = build_gemma_writer(
        config["llm_path"],
        device,
        lora_r=int(writer_config["lora_r"]),
        lora_alpha=int(writer_config["lora_alpha"]),
    )
    semantics = (
        label_semantic_embeddings(llm, tokenizer, label_names)
        if mil_mode == "all_classes"
        else None
    )
    model = ReportWriter(
        llm,
        tokenizer,
        semantics,
        visual_dim=int(config["encoder"]["dim_latent"]),
        num_visual_queries=int(writer_config["num_visual_queries"]),
        resampler_depth=int(writer_config["resampler_depth"]),
        resampler_heads=int(writer_config["resampler_heads"]),
        max_target_tokens=int(writer_config["max_target_tokens"]),
        mil_conditioning=mil_mode,
        llm_dim=hidden_size if mil_mode == "none" else None,
    ).to(device)
    checkpoint = load_writer_checkpoint(args.checkpoint, model, label_names)
    model.eval()

    indices = shard_indices(len(source), args.num_shards, args.shard_index)
    if args.max_studies:
        indices = indices[: int(args.max_studies)]
    max_new_tokens = args.max_new_tokens or int(writer_config["max_target_tokens"])

    output_path = shard_output_path(
        Path(args.output_csv).resolve(), args.num_shards, args.shard_index
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume:
        completed = completed_study_uids(output_path)
    elif not args.overwrite and output_path.exists() and output_path.stat().st_size:
        raise FileExistsError(
            f"{output_path} already exists; pass --resume to continue it or "
            f"--overwrite to replace it"
        )
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_update": int(checkpoint.get("update", -1)),
        "mode": args.mode,
        "split": args.split,
        "studies": len(indices),
        "max_new_tokens": max_new_tokens,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume": bool(args.resume),
        "already_completed": len(completed),
        "decoding": "greedy",
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata), flush=True)

    started = time.time()
    skipped = 0
    mode = "a" if args.resume and completed else "w"
    with output_path.open(mode, newline="") as handle:
        writer = csv.writer(handle)
        if mode == "w":
            writer.writerow(["study_uid", "findings_gt", "findings_pred"])
        for position, index in enumerate(indices):
            subject_id = subject_ids[index]
            if subject_id in completed:
                skipped += 1
                continue
            if args.mode == "cached":
                item = source[index]
                tokens = item["tokens"].to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
            else:
                item = source.get(index)
                tokens = item["tokens"]
            target = targets[subject_id]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mil_logits = (
                    infer_mil(mil_head, tokens) if mil_head is not None else None
                )
                prediction = model.generate(
                    tokens,
                    mil_logits,
                    thresholds,
                    max_new_tokens=max_new_tokens,
                )
            writer.writerow([subject_id, target.text, prediction])
            handle.flush()
            if (position + 1) % 10 == 0 or position + 1 == len(indices):
                elapsed = time.time() - started
                print(
                    f"generated={position + 1 - skipped}/{len(indices)} "
                    f"skipped={skipped} elapsed={elapsed:.0f}s",
                    flush=True,
                )
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
