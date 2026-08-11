from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    project_root = config_path.parent.parent
    for container, key in (
        (config, "output_dir"),
        (config.get("data", {}), "cached_tokens_dir"),
    ):
        if key in container and not Path(container[key]).is_absolute():
            container[key] = str((project_root / container[key]).resolve())
    config["_config_path"] = str(config_path)
    return config


def require_training_policy(config: dict) -> None:
    writer = config["writer"]
    training = config["training"]
    if writer.get("mil_conditioning", "all_classes") not in ("all_classes", "none"):
        raise ValueError(
            "mil_conditioning must be all_classes or none "
            "(none = no-classification-labels ablation)"
        )
    if float(writer.get("mil_proposal_dropout", -1)) != 0.0:
        raise ValueError("MR-RATE strategy fixes MIL proposal dropout at zero")
    if bool(writer.get("localization", True)):
        raise ValueError("MR-RATE strategy has localization completely disabled")
    if bool(training.get("replacement_sampling", True)):
        raise ValueError("Replacement sampling violates exact epoch coverage")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    if int(training.get("batch_size", 0)) != 1:
        raise ValueError("Ragged MR studies require batch_size=1 per GPU")
