from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, require_training_policy
from .mil import load_frozen_mil
from .provenance import verify_mil_encoder_provenance
from .targets import load_target_index


def check(config: dict, mode: str) -> dict:
    require_training_policy(config)
    required = {
        "upstream_root": Path(config["upstream_root"]),
        "encoder_checkpoint": Path(config["encoder_checkpoint"]),
        "mil_checkpoint": Path(config["mil_checkpoint"]),
        "llm_path": Path(config["llm_path"]),
        "jsonl_file": Path(config["data"]["jsonl_file"]),
        "labels_file": Path(config["data"]["labels_file"]),
        "splits_csv": Path(config["data"]["splits_csv"]),
    }
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing configured artifacts: {missing}")
    targets = load_target_index(required["jsonl_file"])
    _, labels, thresholds = load_frozen_mil(
        required["mil_checkpoint"],
        required["upstream_root"],
        expected_dim=int(config["encoder"]["dim_latent"]),
    )
    result = {
        "mode": mode,
        "report_targets": len(targets),
        "report_statements": sum(
            len(value.statements) for value in targets.values()
        ),
        "empty_report_targets": sum(
            not value.statements for value in targets.values()
        ),
        "mil_classes": len(labels),
        "mil_thresholds": int(thresholds.numel()),
    }
    if mode == "cached":
        from .cache import ExactRaggedTokenDataset

        dataset = ExactRaggedTokenDataset(
            config["data"]["cached_tokens_dir"],
            "train",
            targets,
            expected_dim=int(config["encoder"]["dim_latent"]),
            expected_label_names=labels,
        )
        result.update(
            train_studies=len(dataset),
            train_tokens=dataset.num_tokens,
            cache_fingerprint=dataset.metadata.get("cache_fingerprint"),
        )
        result["provenance"] = verify_mil_encoder_provenance(
            required["mil_checkpoint"],
            required["encoder_checkpoint"],
            config["encoder"],
            cache_metadata=dataset.metadata,
        )
    elif mode == "online":
        if config["encoder"]["fusion_mode"] != "late":
            raise ValueError("Online exact token training requires late fusion")
        online_data = (
            config["data"].get("preprocessed_dir")
            if config["data"].get("use_preprocessed")
            else config["data"].get("data_folder")
        )
        if not online_data or not Path(online_data).exists():
            raise FileNotFoundError(f"Missing online MR data source: {online_data}")
        result["train_source"] = "frozen encoder"
        result["provenance"] = verify_mil_encoder_provenance(
            required["mil_checkpoint"],
            required["encoder_checkpoint"],
            config["encoder"],
        )
    else:
        raise ValueError("mode must be online or cached")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("online", "cached"), required=True)
    args = parser.parse_args()
    print(json.dumps(check(load_config(args.config), args.mode), indent=2))


if __name__ == "__main__":
    main()
