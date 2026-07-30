from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mil_encoder_contract(package: dict) -> tuple[dict | None, dict | None]:
    """Return recorded encoder file identity and architecture configuration."""

    online = package.get("data_provenance")
    if isinstance(online, dict):
        return online.get("encoder_checkpoint"), online.get("encoder")
    caches = package.get("cache_provenance")
    if isinstance(caches, dict):
        train = caches.get("train")
        if isinstance(train, dict):
            provenance = train.get("provenance")
            if isinstance(provenance, dict):
                encoder = {
                    "name": provenance.get("encoder"),
                    "chunk_size": provenance.get("chunk_size"),
                    "fusion_mode": provenance.get("fusion_mode"),
                    "pooling_strategy": provenance.get("pooling_strategy"),
                    "dim_latent": provenance.get("dim_latent"),
                    "extra_latent_projection": provenance.get(
                        "extra_latent_projection"
                    ),
                }
                return provenance.get("checkpoint"), encoder
    return None, None


def _compare_config(recorded: dict, configured: dict) -> None:
    configured_contract = {
        "name": configured.get("name"),
        "chunk_size": configured.get("chunk_size"),
        "fusion_mode": configured.get("fusion_mode"),
        "pooling_strategy": configured.get("pooling_strategy"),
        "dim_latent": configured.get("dim_latent"),
        "extra_latent_projection": bool(
            configured.get("extra_latent_projection", False)
        ),
    }
    for key, current in configured_contract.items():
        previous = recorded.get(key)
        if previous is None:
            continue
        if key == "extra_latent_projection":
            previous, current = bool(previous), bool(current)
        if previous != current:
            raise ValueError(
                f"MIL encoder configuration mismatch for {key}: "
                f"checkpoint={previous!r}, configured={current!r}"
            )


def verify_mil_encoder_provenance(
    mil_checkpoint: str | Path,
    encoder_checkpoint: str | Path,
    encoder_config: dict,
    *,
    cache_metadata: dict | None = None,
) -> dict[str, Any]:
    """Prove that MIL, encoder, and optional token cache share one origin."""

    package = torch.load(mil_checkpoint, map_location="cpu", weights_only=False)
    recorded_file, recorded_config = _mil_encoder_contract(package)
    if not isinstance(recorded_file, dict) or not recorded_file.get("sha256"):
        raise ValueError(
            "MIL checkpoint has no verifiable encoder SHA-256 provenance"
        )
    actual_path = Path(encoder_checkpoint).resolve()
    actual_sha = sha256_file(actual_path)
    if actual_sha != recorded_file["sha256"]:
        raise ValueError(
            "MIL checkpoint was trained with a different encoder checkpoint: "
            f"recorded={recorded_file['sha256']}, actual={actual_sha}"
        )
    if isinstance(recorded_config, dict):
        _compare_config(recorded_config, encoder_config)

    cache_fingerprint = None
    if cache_metadata is not None:
        cache_fingerprint = cache_metadata.get("cache_fingerprint")
        if not cache_fingerprint:
            raise ValueError("Current token cache has no verified fingerprint")
        cached_provenance = package.get("cache_provenance")
        if isinstance(cached_provenance, dict):
            train = cached_provenance.get("train")
            recorded_fingerprint = (
                train.get("cache_fingerprint") if isinstance(train, dict) else None
            )
            if not recorded_fingerprint:
                raise ValueError(
                    "Cached MIL checkpoint lacks its training-cache fingerprint"
                )
            if recorded_fingerprint != cache_fingerprint:
                raise ValueError(
                    "Report token cache differs from the cache used to train MIL"
                )
    return {
        "encoder_sha256": actual_sha,
        "encoder_config_verified": isinstance(recorded_config, dict),
        "cache_fingerprint": cache_fingerprint,
        "cache_verified": cache_metadata is not None,
    }

