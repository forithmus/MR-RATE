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
    mil_mode = str(config["writer"].get("mil_conditioning", "all_classes"))
    required = {
        "upstream_root": Path(config["upstream_root"]),
        "encoder_checkpoint": Path(config["encoder_checkpoint"]),
        "llm_path": Path(config["llm_path"]),
        "jsonl_file": Path(config["data"]["jsonl_file"]),
        "reports_csv": Path(config["data"]["reports_csv"]),
        "labels_file": Path(config["data"]["labels_file"]),
        "splits_csv": Path(config["data"]["splits_csv"]),
    }
    if mil_mode == "all_classes":
        required["mil_checkpoint"] = Path(config["mil_checkpoint"])
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing configured artifacts: {missing}")
    targets = load_target_index(required["reports_csv"])
    if mil_mode == "all_classes":
        _, labels, thresholds = load_frozen_mil(
            required["mil_checkpoint"],
            required["upstream_root"],
            expected_dim=int(config["encoder"]["dim_latent"]),
        )
    else:
        labels, thresholds = [], None
    result = {
        "mode": mode,
        "mil_conditioning": mil_mode,
        "report_targets": len(targets),
        "findings_characters": sum(
            len(value.findings) for value in targets.values()
        ),
        "mil_classes": len(labels),
        "mil_thresholds": (
            int(thresholds.numel()) if thresholds is not None else 0
        ),
    }
    if mode == "cached":
        from .cache import ExactRaggedTokenDataset

        dataset = ExactRaggedTokenDataset(
            config["data"]["cached_tokens_dir"],
            "train",
            targets,
            expected_dim=int(config["encoder"]["dim_latent"]),
            expected_label_names=labels if mil_mode == "all_classes" else None,
        )
        result.update(
            train_studies=len(dataset),
            train_tokens=dataset.num_tokens,
            cache_fingerprint=dataset.metadata.get("cache_fingerprint"),
        )
        if mil_mode == "all_classes":
            result["provenance"] = verify_mil_encoder_provenance(
                required["mil_checkpoint"],
                required["encoder_checkpoint"],
                config["encoder"],
                cache_metadata=dataset.metadata,
            )
        else:
            result["provenance"] = {
                "skipped": "mil_conditioning=none has no MIL provenance"
            }
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
        if mil_mode == "all_classes":
            result["provenance"] = verify_mil_encoder_provenance(
                required["mil_checkpoint"],
                required["encoder_checkpoint"],
                config["encoder"],
            )
        else:
            result["provenance"] = {
                "skipped": "mil_conditioning=none has no MIL provenance"
            }
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
