from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def _import_mil_class(upstream_root: str | Path):
    scripts = Path(upstream_root) / "scripts"
    if not (scripts / "mil_probe.py").exists():
        raise FileNotFoundError(f"Missing upstream MIL code: {scripts / 'mil_probe.py'}")
    sys.path.insert(0, str(scripts))
    try:
        from mil_probe import ClassifyThenAggregate
    finally:
        sys.path.pop(0)
    return ClassifyThenAggregate


def load_frozen_mil(
    checkpoint: str | Path,
    upstream_root: str | Path,
    *,
    expected_dim: int = 512,
) -> tuple[nn.Module, list[str], torch.Tensor]:
    package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = package.get("architecture")
    state = package.get("state_dict") or package.get("head_state_dict")
    label_names = [str(value) for value in package.get("label_names", [])]
    if not isinstance(architecture, dict) or not isinstance(state, dict):
        raise ValueError("MIL checkpoint lacks architecture/state_dict")
    dim = int(architecture["dim"])
    classes = int(architecture["n_classes"])
    if dim != int(expected_dim):
        raise ValueError(f"MIL input dim {dim} != expected {expected_dim}")
    if len(label_names) != classes or len(set(label_names)) != classes:
        raise ValueError("MIL checkpoint has an invalid label schema")
    cls = _import_mil_class(upstream_root)
    head = cls(
        dim=dim,
        n_classes=classes,
        hidden_dim=int(architecture.get("hidden_dim", 512)),
        mlp_hidden_dims=tuple(architecture.get("mlp_hidden_dims", [384])),
        drop_rate=float(architecture.get("drop_rate", 0.0)),
        init_std=float(architecture.get("init_std", 0.02)),
        use_gating=bool(architecture.get("use_gating", True)),
        use_norm=bool(architecture.get("use_norm", False)),
        use_output_bias_scale=bool(
            architecture.get("use_output_bias_scale", True)
        ),
    )
    head.load_state_dict(state, strict=True)
    loaded = head.state_dict()
    for name, expected in state.items():
        actual = loaded[name].detach().cpu()
        expected = expected.detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"MIL tensor changed while loading: {name}")
        if actual.is_floating_point() and not torch.isfinite(actual).all():
            raise ValueError(f"MIL checkpoint contains non-finite tensor: {name}")
    head.requires_grad_(False)
    head.eval()
    thresholds = torch.as_tensor(
        package.get("validation_thresholds", [0.0] * classes),
        dtype=torch.float32,
    )
    if thresholds.shape != (classes,) or not torch.isfinite(thresholds).all():
        raise ValueError("MIL validation thresholds are invalid")
    # The upstream selector stores thresholds in logit space. Conditioning
    # compares sigmoid probabilities, so convert the boundary exactly once.
    return head, label_names, thresholds.sigmoid()


@torch.no_grad()
def infer_mil(head: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    cu_seqlens = torch.tensor(
        [0, tokens.shape[0]], dtype=torch.long, device=tokens.device
    )
    logits = head(tokens, cu_seqlens)
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected MIL output: {tuple(logits.shape)}")
    return logits.float()
