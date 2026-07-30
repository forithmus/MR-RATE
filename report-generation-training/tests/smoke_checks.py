"""Dependency-light single-writer checks runnable without pytest."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mrrate_report_training.cache import ExactRaggedTokenDataset
from mrrate_report_training.mil import load_frozen_mil
from mrrate_report_training.model import ReportWriter
from mrrate_report_training.provenance import verify_mil_encoder_provenance
from mrrate_report_training.targets import make_report_target
from mrrate_report_training.train import exact_rank_indices


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text, **_):
        values = [2 + (ord(char) % 13) for char in text][:20] or [2]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16)
        self.lora_report = nn.Parameter(torch.tensor(0.1))
        self.active = "report"
        for value in (*self.embedding.parameters(), *self.output.parameters()):
            value.requires_grad_(False)

    def set_adapter(self, value):
        self.active = value

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, logits_to_keep, **_):
        return SimpleNamespace(
            logits=self.output(
                inputs_embeds[:, -logits_to_keep:] + self.lora_report
            )
        )


def check_targets() -> None:
    statements = [
        "There is no acute intracranial abnormality.",
        "A chronic infarct is present.",
        "Cannot exclude hemorrhage.",
    ]
    target = make_report_target("s", statements)
    assert target.statements == tuple(statements)
    assert target.text == " ".join(statements)
    assert not hasattr(target, "abnormal")
    assert not hasattr(target, "normal")


def check_coverage() -> None:
    shards = [
        exact_rank_indices(11, 0, 7, 4, rank, True) for rank in range(4)
    ]
    real = [value for shard in shards for value in shard if value >= 0]
    assert sorted(real) == list(range(11))
    assert len(real) == len(set(real))


def check_cache_and_path_equivalence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokens = np.arange(30, dtype=np.float16).reshape(6, 5)
        tokens.tofile(root / "tokens.bin")
        np.save(root / "offsets.npy", [0, 2, 6])
        np.save(root / "labels.npy", [[1, 0], [0, 1]])
        np.save(root / "full.npy", [2, 4])
        np.save(root / "series.npy", [1, 2])
        (root / "ids.txt").write_text("a\nb\n")
        (root / "label_names.json").write_text(json.dumps(["x", "y"]))
        (root / "token_features_train.json").write_text(
            json.dumps(
                {
                    "format": "raw_numpy_memmap",
                    "feature_level": "projected_per_series_visual_tokens",
                    "max_tokens_per_study": 0,
                    "dim": 5,
                    "dtype": "float16",
                    "tokens_file": "tokens.bin",
                    "offsets_file": "offsets.npy",
                    "labels_file": "labels.npy",
                    "subject_ids_file": "ids.txt",
                    "full_token_counts_file": "full.npy",
                    "series_counts_file": "series.npy",
                }
            )
        )
        targets = {
            value: make_report_target(
                value,
                ["There is no abnormality.", "A chronic finding is present."],
            )
            for value in ("a", "b")
        }
        cache = ExactRaggedTokenDataset(
            root,
            "train",
            targets,
            expected_dim=5,
            expected_label_names=["x", "y"],
        )
        online = torch.from_numpy(tokens[:2].copy())
        cached = cache[0]["tokens"]
        assert torch.equal(cached, online)
        model = ReportWriter(
            TinyLLM(),
            TinyTokenizer(),
            torch.randn(2, 8),
            visual_dim=5,
            num_visual_queries=2,
            resampler_depth=1,
            resampler_heads=1,
            max_target_tokens=20,
        ).eval()
        logits = torch.tensor([[0.2, -0.7]])
        thresholds = torch.tensor([0.5, 0.5])
        with torch.no_grad():
            online_losses = model(online, logits, thresholds, targets["a"])
            cached_losses = model(cached, logits, thresholds, targets["a"])
        for name in ("loss", "report_loss"):
            assert torch.equal(online_losses[name], cached_losses[name]), name


def check_single_writer_update() -> None:
    llm = TinyLLM()
    model = ReportWriter(
        llm,
        TinyTokenizer(),
        torch.randn(3, 8),
        visual_dim=8,
        num_visual_queries=2,
        resampler_depth=1,
        resampler_heads=2,
        max_target_tokens=20,
    )
    calls = []
    hook = model.resampler.register_forward_hook(lambda *_: calls.append(1))
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=1e-3
    )
    before = llm.lora_report.detach().clone()
    losses = model(
        torch.randn(5, 8),
        torch.randn(1, 3),
        torch.full((3,), 0.5),
        make_report_target(
            "s", ["There is no hemorrhage.", "A small infarct is present."]
        ),
    )
    losses["loss"].backward()
    hook.remove()
    assert len(calls) == 1
    assert llm.lora_report.grad is not None
    optimizer.step()
    assert not torch.equal(before, llm.lora_report)
    lora_names = [name for name, _ in llm.named_parameters() if "lora_" in name]
    assert lora_names == ["lora_report"]


def check_strict_weight_and_provenance_loading() -> None:
    upstream = Path(
        "/hnvme/workspace/b180dc51-sezgin/"
        "MR-RATE-linearprobe/contrastive-pretraining"
    )
    sys.path.insert(0, str(upstream / "scripts"))
    from mil_probe import ClassifyThenAggregate

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        encoder = root / "encoder.pt"
        encoder.write_bytes(b"synthetic encoder identity")
        encoder_sha = hashlib.sha256(encoder.read_bytes()).hexdigest()
        head = ClassifyThenAggregate(
            dim=8,
            n_classes=2,
            hidden_dim=4,
            mlp_hidden_dims=(3,),
            drop_rate=0.0,
            use_gating=True,
            use_norm=False,
            use_output_bias_scale=True,
        )
        checkpoint = root / "mil_head.pt"
        architecture = {
            "dim": 8,
            "n_classes": 2,
            "hidden_dim": 4,
            "mlp_hidden_dims": [3],
            "drop_rate": 0.0,
            "use_gating": True,
            "use_norm": False,
            "use_output_bias_scale": True,
        }
        encoder_config = {
            "name": "vjepa2",
            "chunk_size": 64,
            "fusion_mode": "late",
            "pooling_strategy": "simple_attn",
            "dim_latent": 8,
            "extra_latent_projection": False,
        }
        torch.save(
            {
                "state_dict": head.state_dict(),
                "architecture": architecture,
                "label_names": ["a", "b"],
                "validation_thresholds": [0.0, 0.0],
                "data_provenance": {
                    "encoder_checkpoint": {"sha256": encoder_sha},
                    "encoder": encoder_config,
                },
            },
            checkpoint,
        )
        loaded, labels, thresholds = load_frozen_mil(
            checkpoint, upstream, expected_dim=8
        )
        assert labels == ["a", "b"]
        assert torch.equal(thresholds, torch.full((2,), 0.5))
        for name, value in head.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[name])
        verified = verify_mil_encoder_provenance(
            checkpoint, encoder, encoder_config
        )
        assert verified["encoder_sha256"] == encoder_sha
        encoder.write_bytes(b"wrong encoder")
        try:
            verify_mil_encoder_provenance(checkpoint, encoder, encoder_config)
        except ValueError as error:
            assert "different encoder checkpoint" in str(error)
        else:
            raise AssertionError("Mismatched encoder provenance was accepted")


if __name__ == "__main__":
    check_targets()
    check_coverage()
    check_cache_and_path_equivalence()
    check_single_writer_update()
    check_strict_weight_and_provenance_loading()
    print("single-writer smoke checks: PASS")

