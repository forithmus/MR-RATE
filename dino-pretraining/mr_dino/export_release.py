"""Export the EMA-teacher backbone of a DCP training checkpoint as an inference release.

The release is the directory ``mr_dino.release.load_release`` reads (and the MIL code takes as
``--release``): ``config.json``, ``teacher_backbone.bf16.pt``, the model source
``mr_dino/model.py`` and ``training/checkpoint_metadata.json``. A modality-channel backbone
also records its channel layout under ``input_channels``.

    python -m mr_dino.export_release --checkpoint <run>/checkpoints/step_00050000 --output <release dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from .channels import MODALITIES
from .data import stage_crop_spec
from .probe import load_teacher_backbone

FORMAT = "mrdino3d_teacher_backbone_bf16_v1"


def release_config(meta: dict, backbone) -> dict:
    args = meta["args"]
    spec = stage_crop_spec(args["stage"] if "stage" in args else meta["stage"])
    global_shape = list(args.get("global_shape") or spec.global_shape)
    local_shape = list(args.get("local_shape") or spec.local_shape)
    config = {
        "format": FORMAT,
        "stage": meta["stage"],
        "step": int(meta["step"]),
        "source": "model.teacher.backbone",
        "arch": args["arch"],
        "params_billion": round(sum(p.numel() for p in backbone.parameters()) / 1e9, 3),
        "space": args["space"],
        "weights": "teacher_backbone.bf16.pt",
        "backbone": {
            "volume_size": list(backbone.patch_embed.volume_size),
            "patch_size": list(backbone.patch_embed.patch_size),
            "voxel_spacing_mm": list(args["target_spacing"]),
            "in_chans": int(backbone.patch_embed.in_chans),
            "n_storage_tokens": int(backbone.n_storage_tokens),
            "drop_path_rate": 0.4 if args["arch"] == "7b" else (0.3 if args["arch"] != "tiny" else 0.0),
            "layerscale_init": 1e-5,
        },
        "training_crops": {"global": global_shape, "local": local_shape},
    }
    if args.get("input_channels", "sequence") == "modality":
        config["input_channels"] = {"layout": "modality", "modalities": list(MODALITIES), "availability_planes": True}
    return config


def export_release(checkpoint: Path, output: Path) -> Path:
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new release directory")
    backbone, meta = load_teacher_backbone(checkpoint)
    tmp = output.with_name(output.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "mr_dino").mkdir(parents=True)
    (tmp / "training").mkdir()
    config = release_config(meta, backbone)
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in backbone.state_dict().items()}
    torch.save(state, tmp / config["weights"])
    shutil.copy2(Path(__file__).with_name("model.py"), tmp / "mr_dino" / "model.py")
    (tmp / "training" / "checkpoint_metadata.json").write_text(
        json.dumps({"args": meta["args"], "step": int(meta["step"]), "stage": meta["stage"],
                    "checkpoint": str(checkpoint.resolve())}, indent=2, sort_keys=True))
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    os.replace(tmp, output)
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="DCP checkpoint directory (step_XXXXXXXX)")
    p.add_argument("--output", required=True, help="new release directory")
    args = p.parse_args(argv)
    print(export_release(Path(args.checkpoint), Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
