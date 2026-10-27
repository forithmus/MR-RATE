"""Load Dong's inference releases without confusing them with DCP resumes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def release_metadata(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("source") != "model.teacher.backbone":
        raise ValueError("Expected an EMA teacher backbone release")
    provenance = "release training metadata"
    checkpoint_metadata = path / "training/checkpoint_metadata.json"
    if checkpoint_metadata.is_file():
        args = json.loads(checkpoint_metadata.read_text())["args"]
    else:
        candidates = [path / "training/run_config.json",
                      path / f"training/run_config_{config['stage']}.json"]
        found = next((p for p in candidates if p.is_file()), None)
        if found is not None:
            args = json.loads(found.read_text())
        elif config.get("format")=="mrdino3d_7b_teacher_backbone_bf16_v1" and config.get("mr_rate_commit")=="43c05c6":
            # This legacy release omitted the run arguments. Its README states
            # atlas space; the pinned trainer below supplies these defaults.
            # NVIDIA can supply the original overrides in preprocessing.json.
            config.setdefault("space","atlas_space")
            config.setdefault("training_crops",{"global":config["backbone"]["volume_size"]})
            args={"arch":config["arch"],"space":"atlas_space","target_shape":[256,384,384],
                  "target_spacing":[1.,.5,.5],"posterior_shift_mm":15.}
            provenance="legacy 7B README + pinned 43c05c61bf98d00c4e825a27a0fca3c742754d78 trainer defaults; original run overrides absent"
            if (path/"preprocessing.json").is_file():
                args.update(json.loads((path/"preprocessing.json").read_text()))
                provenance="legacy release + supplied preprocessing.json"
        else:
            raise ValueError("Release has no training preprocessing metadata; supply its training configuration")
    for key in ("arch", "space"):
        if args.get(key) != config[key]:
            raise ValueError(f"Release/training metadata disagree on {key}")
    if tuple(args["target_spacing"]) != tuple(config["backbone"]["voxel_spacing_mm"]):
        raise ValueError("Release/training voxel spacing mismatch")
    weights = (path / config["weights"]).resolve()
    if not weights.is_relative_to(path):
        raise ValueError("Weights must belong to the release directory")
    return {"step": config["step"], "stage": config["stage"], "args": args,
            "source": str(path), "config": config,"preprocessing_provenance":provenance}


def load_release(path: str | Path, device="cpu"):
    """Use the supplied model source, strict keys, and memory-mapped BF16 weights."""
    path = Path(path).resolve()
    metadata = release_metadata(path)
    config = metadata["config"]
    spec = importlib.util.spec_from_file_location("dong_release_model", path / "mr_dino/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constructor = getattr(module, f"vit_{config['arch']}_3d")
    with torch.device("meta"):
        backbone = constructor(**config["backbone"])
    state = torch.load(path / config["weights"], map_location="cpu", weights_only=True, mmap=True)
    backbone.load_state_dict(state, strict=True, assign=True)
    backbone.to(device=device)
    backbone.eval().requires_grad_(False)
    return backbone, metadata
