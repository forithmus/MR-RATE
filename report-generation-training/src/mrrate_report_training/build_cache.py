from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .online import OnlineSource


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cache_provenance(config: dict) -> tuple[dict, str]:
    checkpoint = Path(config["encoder_checkpoint"]).resolve()
    stat = checkpoint.stat()
    encoder = config["encoder"]
    data = config["data"]
    provenance = {
        "version": 1,
        "checkpoint": {
            "path": str(checkpoint),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "encoder": encoder,
        "jsonl_file": str(Path(data["jsonl_file"]).resolve()),
        "labels_file": str(Path(data["labels_file"]).resolve()),
        "splits_csv": str(Path(data["splits_csv"]).resolve()),
        "data_folder": (
            str(Path(data["data_folder"]).resolve()) if data.get("data_folder") else None
        ),
        "preprocessed_dir": (
            str(Path(data["preprocessed_dir"]).resolve())
            if data.get("preprocessed_dir")
            else None
        ),
        "use_preprocessed": bool(data.get("use_preprocessed", False)),
        "normalizer": data.get("normalizer", "zscore"),
        "space": data.get("space", "native_space"),
        "cache_dtype": "float16",
        "max_tokens_per_study": 0,
    }
    serialized = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return provenance, hashlib.sha256(serialized.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Exact cache generation requires a CUDA GPU")
    torch.cuda.set_device(0)
    source = OnlineSource(config, torch.device("cuda", 0), split=args.split)
    out = Path(config["data"]["cached_tokens_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance, fingerprint = cache_provenance(config)
    for manifest_path in out.glob("token_features_*.json"):
        existing = json.loads(manifest_path.read_text())
        if existing.get("cache_fingerprint") != fingerprint:
            raise ValueError(
                f"{manifest_path} was built with a different encoder/preprocessing"
            )

    cache_id = uuid.uuid4().hex[:12]
    token_path = out / f"tokens_{args.split}_{cache_id}.bin"
    token_temporary = token_path.with_name(token_path.name + f".tmp.{os.getpid()}")
    offsets = [0]
    full_counts, series_counts, labels, subject_ids = [], [], [], []
    try:
        with token_temporary.open("wb") as handle:
            for index in range(len(source)):
                item = source.get(index)
                array = item["tokens"].float().cpu().numpy().astype(
                    np.float16, copy=False
                )
                if not len(array):
                    raise ValueError(f"{item['subject_id']} produced an empty token bag")
                array.tofile(handle)
                offsets.append(offsets[-1] + len(array))
                full_counts.append(len(array))
                series_counts.append(item["series_count"])
                labels.append(item["mil_labels"].numpy().astype(np.float32))
                subject_ids.append(item["subject_id"])
                if (index + 1) % 100 == 0:
                    print(
                        f"encoded={index + 1}/{len(source)} tokens={offsets[-1]}",
                        flush=True,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(token_temporary, token_path)
    except BaseException:
        token_temporary.unlink(missing_ok=True)
        raise

    offsets_path = out / f"token_offsets_{args.split}_{cache_id}.npy"
    labels_path = out / f"labels_{args.split}_{cache_id}.npy"
    ids_path = out / f"subject_ids_{args.split}_{cache_id}.txt"
    full_path = out / f"full_token_counts_{args.split}_{cache_id}.npy"
    series_path = out / f"series_counts_{args.split}_{cache_id}.npy"
    atomic_npy(offsets_path, np.asarray(offsets, dtype=np.int64))
    atomic_npy(labels_path, np.stack(labels).astype(np.float32))
    atomic_npy(full_path, np.asarray(full_counts, dtype=np.int64))
    atomic_npy(series_path, np.asarray(series_counts, dtype=np.int32))
    atomic_text(ids_path, "\n".join(subject_ids) + "\n")
    label_names = [str(value) for value in source.dataset.label_columns]
    names_path = out / "label_names.json"
    if names_path.exists() and json.loads(names_path.read_text()) != label_names:
        raise ValueError("Existing cache label_names.json differs")
    atomic_text(names_path, json.dumps(label_names, indent=2) + "\n")
    manifest = {
        "format": "raw_numpy_memmap",
        "format_version": 2,
        "feature_level": "projected_per_series_visual_tokens",
        "split": args.split,
        "tokens_file": token_path.name,
        "offsets_file": offsets_path.name,
        "labels_file": labels_path.name,
        "subject_ids_file": ids_path.name,
        "full_token_counts_file": full_path.name,
        "series_counts_file": series_path.name,
        "dtype": "float16",
        "dim": int(config["encoder"]["dim_latent"]),
        "num_studies": len(subject_ids),
        "num_tokens": offsets[-1],
        "max_tokens_per_study": 0,
        "cache_fingerprint": fingerprint,
        "provenance": provenance,
    }
    # Publish the canonical manifest last.
    atomic_text(
        out / f"token_features_{args.split}.json",
        json.dumps(manifest, indent=2) + "\n",
    )
    print(
        f"exact cache complete: studies={len(subject_ids)} tokens={offsets[-1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()

