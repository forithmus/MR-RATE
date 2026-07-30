"""One-GPU end-to-end integration gate using a deterministic dummy MR dataset."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mrrate_report_training.cache import ExactRaggedTokenDataset
from mrrate_report_training.mil import infer_mil
from mrrate_report_training.model import ReportWriter
from mrrate_report_training.targets import make_report_target
from mrrate_report_training.train import (
    cosine_schedule,
    load_checkpoint,
    save_checkpoint,
)


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        maximum = int(kwargs.get("max_length", 128))
        values = [2 + (ord(char) % 61) for char in text][:maximum] or [2]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))


class TinyReportAdapterLLM(nn.Module):
    """Small causal decoder surface with one actual low-rank report adapter."""

    def __init__(self, hidden: int = 32, vocabulary: int = 64, rank: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.output = nn.Linear(hidden, vocabulary)
        self.lora_A_report = nn.Parameter(torch.randn(hidden, rank) * 0.02)
        self.lora_B_report = nn.Parameter(torch.zeros(rank, hidden))
        self.active_adapter = "report"
        self.embedding.requires_grad_(False)
        self.output.requires_grad_(False)

    def set_adapter(self, name: str) -> None:
        if str(name) != "report":
            raise ValueError(f"unexpected adapter: {name}")
        self.active_adapter = str(name)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, logits_to_keep, **_):
        hidden = inputs_embeds[:, -int(logits_to_keep) :]
        hidden = hidden + hidden @ self.lora_A_report @ self.lora_B_report
        return SimpleNamespace(logits=self.output(hidden))


def write_dummy_cache(root: Path) -> tuple[dict, list[torch.Tensor], list[str]]:
    generator = np.random.default_rng(41)
    subject_ids = [f"dummy_{index:02d}" for index in range(4)]
    token_counts = [257, 129, 301, 193]
    bags = [
        generator.normal(size=(count, 512)).astype(np.float16)
        for count in token_counts
    ]
    labels = generator.integers(0, 2, size=(len(bags), 74)).astype(np.float32)
    label_names = [f"synthetic_mr_finding_{index:02d}" for index in range(74)]
    root.mkdir(parents=True, exist_ok=True)
    tokens_path = root / "tokens_train_dummy.bin"
    with tokens_path.open("wb") as handle:
        for bag in bags:
            bag.tofile(handle)
    offsets = np.concatenate(([0], np.cumsum(token_counts))).astype(np.int64)
    np.save(root / "offsets.npy", offsets)
    np.save(root / "labels.npy", labels)
    np.save(root / "full_counts.npy", np.asarray(token_counts, dtype=np.int64))
    np.save(root / "series_counts.npy", np.ones(len(bags), dtype=np.int32))
    (root / "subject_ids.txt").write_text("\n".join(subject_ids) + "\n")
    (root / "label_names.json").write_text(json.dumps(label_names))
    (root / "token_features_train.json").write_text(
        json.dumps(
            {
                "format": "raw_numpy_memmap",
                "format_version": 2,
                "feature_level": "projected_per_series_visual_tokens",
                "split": "train",
                "tokens_file": tokens_path.name,
                "offsets_file": "offsets.npy",
                "labels_file": "labels.npy",
                "subject_ids_file": "subject_ids.txt",
                "full_token_counts_file": "full_counts.npy",
                "series_counts_file": "series_counts.npy",
                "dtype": "float16",
                "dim": 512,
                "num_studies": len(bags),
                "num_tokens": int(offsets[-1]),
                "max_tokens_per_study": 0,
                "cache_fingerprint": "synthetic_gpu_e2e_v1",
            },
            indent=2,
        )
    )
    targets = {
        subject_ids[0]: make_report_target(
            subject_ids[0],
            ["There is no hemorrhage.", "A small chronic infarct is present."],
        ),
        subject_ids[1]: make_report_target(
            subject_ids[1], ["There is no acute intracranial abnormality."]
        ),
        subject_ids[2]: make_report_target(
            subject_ids[2],
            ["Possible demyelinating lesion is present.", "The ventricles are normal."],
        ),
        subject_ids[3]: make_report_target(
            subject_ids[3],
            ["Postoperative change is present.", "There is no hydrocephalus."],
        ),
    }
    return targets, [torch.from_numpy(value.copy()) for value in bags], label_names


def build_mil(upstream_root: Path, device: torch.device) -> nn.Module:
    scripts = upstream_root / "scripts"
    sys.path.insert(0, str(scripts))
    from mil_probe import ClassifyThenAggregate

    torch.manual_seed(9)
    head = ClassifyThenAggregate(
        dim=512,
        n_classes=74,
        hidden_dim=64,
        mlp_hidden_dims=(48,),
        drop_rate=0.0,
        use_gating=True,
        use_norm=False,
        use_output_bias_scale=True,
    ).to(device)
    head.requires_grad_(False)
    head.eval()
    return head


def build_writer(device: torch.device) -> ReportWriter:
    torch.manual_seed(13)
    llm = TinyReportAdapterLLM()
    writer = ReportWriter(
        llm,
        TinyTokenizer(),
        torch.randn(74, 32),
        visual_dim=512,
        num_visual_queries=512,
        resampler_depth=2,
        resampler_heads=8,
        max_target_tokens=96,
    )
    return writer.to(device=device, dtype=torch.bfloat16)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This integration gate must run on a Slurm GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    project = Path(__file__).resolve().parents[1]
    output = project / "runs" / f"synthetic_gpu_e2e_{job_id}"
    cache_root = output / "dummy_cache"
    targets, online_bags, label_names = write_dummy_cache(cache_root)
    cached = ExactRaggedTokenDataset(
        cache_root,
        "train",
        targets,
        expected_dim=512,
        expected_label_names=label_names,
    )
    mil = build_mil(
        Path(
            "/hnvme/workspace/b180dc51-sezgin/"
            "MR-RATE-linearprobe/contrastive-pretraining"
        ),
        device,
    )
    thresholds = torch.full((74,), 0.5, device=device)
    writer = build_writer(device)

    # Exact online/cached equivalence through MIL, resampler, and both writers.
    writer.eval()
    online_tokens = online_bags[0].to(device)
    cached_tokens = cached[0]["tokens"].to(device)
    assert torch.equal(online_tokens, cached_tokens)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        online_logits = infer_mil(mil, online_tokens)
        cached_logits = infer_mil(mil, cached_tokens)
        assert torch.equal(online_logits, cached_logits)
        online_loss = writer(
            online_tokens, online_logits, thresholds, targets["dummy_00"]
        )
        cached_loss = writer(
            cached_tokens, cached_logits, thresholds, targets["dummy_00"]
        )
    for name in ("loss", "report_loss"):
        torch.testing.assert_close(
            online_loss[name], cached_loss[name], rtol=0.0, atol=0.0
        )

    # Real CUDA optimizer steps using the cached path.
    writer.train()
    trainable = [value for value in writer.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.0)
    scheduler = cosine_schedule(optimizer, total_updates=8, warmup_ratio=0.0)
    adapter_before = writer.llm.lora_B_report.detach().clone()
    losses_seen = []
    for index in range(len(cached)):
        item = cached[index]
        tokens = item["tokens"].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = infer_mil(mil, tokens)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = writer(tokens, logits, thresholds, item["target"])
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        scheduler.step()
        losses_seen.append(float(losses["loss"].detach()))
    assert not torch.equal(adapter_before, writer.llm.lora_B_report)
    assert all(parameter.grad is None for parameter in mil.parameters())

    checkpoint = output / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        writer,
        optimizer,
        scheduler,
        {"synthetic": True},
        label_names,
        epoch=0,
        next_slot=len(cached),
        update=len(cached),
        rank=0,
        world=1,
    )

    # Fresh model/optimizer, exact resume, then one more update.
    resumed = build_writer(device)
    resumed_trainable = [
        value for value in resumed.parameters() if value.requires_grad
    ]
    resumed_optimizer = torch.optim.AdamW(
        resumed_trainable, lr=1e-2, weight_decay=0.0
    )
    resumed_scheduler = cosine_schedule(
        resumed_optimizer, total_updates=8, warmup_ratio=0.0
    )
    epoch, slot, update = load_checkpoint(
        str(checkpoint),
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        label_names,
        rank=0,
    )
    assert (epoch, slot, update) == (0, len(cached), len(cached))
    item = cached[0]
    tokens = item["tokens"].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = infer_mil(mil, tokens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        resumed_losses = resumed(tokens, logits, thresholds, item["target"])
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_losses["loss"].backward()
    resumed_optimizer.step()
    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "studies": len(cached),
        "tokens": cached.num_tokens,
        "visual_queries": 512,
        "mil_classes": 74,
        "online_cached_exact": True,
        "optimizer_updates_before_resume": len(cached),
        "post_resume_update": True,
        "report_adapter_updated": True,
        "losses": losses_seen,
        "resumed_loss": float(resumed_losses["loss"].detach()),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "checkpoint": str(checkpoint),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MRRATE_SYNTHETIC_GPU_E2E_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
