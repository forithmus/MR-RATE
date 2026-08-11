"""Tests for the no-classification-labels ablation (mil_conditioning=none)."""

import pytest
import torch

from mrrate_report_training.config import require_training_policy
from mrrate_report_training.generate import load_writer_checkpoint
from mrrate_report_training.model import ReportWriter, trainable_state_dict
from mrrate_report_training.targets import make_report_target
from test_generate import TinyLLM, TinyTokenizer, study_inputs


def build_ablation_writer() -> ReportWriter:
    torch.manual_seed(7)
    return ReportWriter(
        TinyLLM(),
        TinyTokenizer(),
        None,
        visual_dim=8,
        num_visual_queries=2,
        resampler_depth=1,
        resampler_heads=2,
        max_target_tokens=24,
        mil_conditioning="none",
        llm_dim=8,
    )


def build_full_writer() -> ReportWriter:
    torch.manual_seed(7)
    return ReportWriter(
        TinyLLM(),
        TinyTokenizer(),
        torch.randn(3, 8),
        visual_dim=8,
        num_visual_queries=2,
        resampler_depth=1,
        resampler_heads=2,
        max_target_tokens=24,
    )


def test_ablation_prefix_has_no_mil_tokens():
    writer = build_ablation_writer().eval()
    tokens, _, _ = study_inputs()
    visual_prefix, mil_tokens = writer.shared_prefix(tokens, None, None)
    # image_start + num_visual_queries + image_end, and zero MIL tokens.
    assert visual_prefix.shape[1] == 2 + 2
    assert mil_tokens.shape == (1, 0, 8)


def test_ablation_forward_and_generate_without_mil():
    writer = build_ablation_writer()
    tokens, _, _ = study_inputs()
    target = make_report_target("x", "There is no hemorrhage.")
    losses = writer(tokens, None, None, target)
    assert torch.isfinite(losses["loss"])
    writer.eval()
    first = writer.generate(tokens, None, None, max_new_tokens=6)
    assert first == writer.generate(tokens, None, None, max_new_tokens=6)


def test_mode_mixups_raise():
    ablation = build_ablation_writer().eval()
    full = build_full_writer().eval()
    tokens, mil_logits, thresholds = study_inputs()
    with pytest.raises(ValueError, match="received MIL conditioning"):
        ablation.shared_prefix(tokens, mil_logits, thresholds)
    with pytest.raises(ValueError, match="requires MIL logits"):
        full.shared_prefix(tokens, None, None)
    with pytest.raises(ValueError, match="requires label embeddings"):
        ReportWriter(TinyLLM(), TinyTokenizer(), None, visual_dim=8, llm_dim=8)
    with pytest.raises(ValueError, match="must not receive label embeddings"):
        ReportWriter(
            TinyLLM(),
            TinyTokenizer(),
            torch.randn(3, 8),
            visual_dim=8,
            mil_conditioning="none",
        )
    with pytest.raises(ValueError, match="explicit llm_dim"):
        ReportWriter(
            TinyLLM(), TinyTokenizer(), None, visual_dim=8, mil_conditioning="none"
        )


def test_ablation_state_dict_has_no_mil_tensors():
    state = trainable_state_dict(build_ablation_writer())
    assert not any(
        "label_embeddings" in name or "mil_" in name for name in state
    )
    full_state = trainable_state_dict(build_full_writer())
    assert any("mil_value_projection" in name for name in full_state)
    assert "label_embeddings" in full_state


def test_cross_mode_checkpoints_are_refused(tmp_path):
    ablation = build_ablation_writer()
    package = {
        "trainable_state_dict": trainable_state_dict(ablation),
        "label_names": [],
        "config": {"writer": {"mil_conditioning": "none"}},
    }
    path = tmp_path / "ablation.pt"
    torch.save(package, path)
    # Same-mode load works.
    load_writer_checkpoint(path, build_ablation_writer(), [])
    # Into a full-conditioning model: refused by the mode stamp.
    with pytest.raises(ValueError, match="mil_conditioning"):
        load_writer_checkpoint(path, build_full_writer(), ["a", "b", "c"])
    # A legacy full-mode checkpoint (no config) into an ablation model:
    # refused because the stored default mode is all_classes.
    full = build_full_writer()
    legacy = tmp_path / "full.pt"
    torch.save(
        {
            "trainable_state_dict": trainable_state_dict(full),
            "label_names": ["a", "b", "c"],
        },
        legacy,
    )
    with pytest.raises(ValueError, match="mil_conditioning"):
        load_writer_checkpoint(legacy, build_ablation_writer(), [])


def test_policy_validates_mil_conditioning():
    config = {
        "writer": {
            "mil_conditioning": "some_classes",
            "mil_proposal_dropout": 0.0,
            "localization": False,
        },
        "training": {
            "replacement_sampling": False,
            "epochs": 1,
            "batch_size": 1,
        },
    }
    with pytest.raises(ValueError, match="mil_conditioning"):
        require_training_policy(config)
    config["writer"]["mil_conditioning"] = "none"
    require_training_policy(config)
