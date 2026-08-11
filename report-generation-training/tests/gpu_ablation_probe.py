"""GPU probe for the mil_conditioning=none ablation on real Gemma.

Reuses the artifact workspace from tests/gpu_full_stack_e2e.py
(MRRATE_FULLSTACK_DIR): writes an ablation config (no MIL), builds the real
Gemma writer in none mode, takes one optimizer step, saves a trainer-format
checkpoint, reloads it through the inference loader, verifies the
full-conditioning checkpoint is refused cross-mode, and finally runs the
real generate.py CLI in ablation mode over the val split.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from gpu_full_stack_e2e import DIM, SPLITS  # noqa: E402
from mrrate_report_training.cache import ExactRaggedTokenDataset  # noqa: E402
from mrrate_report_training.generate import load_writer_checkpoint  # noqa: E402
from mrrate_report_training.model import (  # noqa: E402
    ReportWriter,
    build_gemma_writer,
    trainable_state_dict,
)
from mrrate_report_training.targets import load_target_index  # noqa: E402
from mrrate_report_training.train import cosine_schedule, save_checkpoint  # noqa: E402

GEMMA_PATH = os.environ.get(
    "MRRATE_LLM_PATH",
    "/hnvme/data/LLM-TZ/hub/models--google--gemma-3-12b-it/snapshots/"
    "96b6f1eccf38110c56df3a15bffe176da04bfd80",
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    workspace = Path(os.environ["MRRATE_FULLSTACK_DIR"])

    base_config = yaml.safe_load((workspace / "config.yaml").read_text())
    base_config["writer"]["mil_conditioning"] = "none"
    base_config["mil_checkpoint"] = "/placeholder/unused/mil_head.pt"
    base_config["output_dir"] = str(workspace / "ablation_runs")
    config_path = workspace / "config_ablation.yaml"
    config_path.write_text(yaml.safe_dump(base_config))

    print("[ablation] loading Gemma writer without MIL conditioning", flush=True)
    started = time.time()
    llm, tokenizer, hidden_size = build_gemma_writer(
        GEMMA_PATH, device, lora_r=16, lora_alpha=32
    )
    writer = ReportWriter(
        llm,
        tokenizer,
        None,
        visual_dim=DIM,
        num_visual_queries=512,
        resampler_depth=2,
        resampler_heads=8,
        max_target_tokens=1536,
        mil_conditioning="none",
        llm_dim=hidden_size,
    ).to(device)
    print(f"[ablation] gemma loaded in {time.time() - started:.0f}s", flush=True)
    state = trainable_state_dict(writer)
    assert not any("label_embeddings" in name or "mil_" in name for name in state)

    targets = load_target_index(workspace / "all_reports.csv")
    dataset = ExactRaggedTokenDataset(
        workspace / "exact_tokens", "val", targets, expected_dim=DIM
    )
    trainable = [value for value in writer.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    scheduler = cosine_schedule(optimizer, total_updates=2, warmup_ratio=0.0)
    writer.train()
    item = dataset[0]
    tokens = item["tokens"].to(device=device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = writer(tokens, None, None, item["target"])
    losses["loss"].backward()
    optimizer.step()
    scheduler.step()
    train_loss = float(losses["loss"].detach())
    print(f"[ablation] one training update done, loss={train_loss:.4f}", flush=True)

    checkpoint_path = workspace / "ablation_last.pt"
    save_checkpoint(
        checkpoint_path,
        writer,
        optimizer,
        scheduler,
        {"writer": {"mil_conditioning": "none"}},
        [],
        epoch=0,
        next_slot=1,
        update=1,
        rank=0,
        world=1,
    )
    load_writer_checkpoint(checkpoint_path, writer, [])
    # The full-conditioning checkpoint from the main run must be refused.
    try:
        load_writer_checkpoint(workspace / "last.pt", writer, [])
    except ValueError as error:
        assert "mil_conditioning" in str(error) or "trainable tensors" in str(error)
        cross_mode_refused = True
    else:
        raise AssertionError("full-conditioning checkpoint loaded into ablation model")
    del writer, optimizer, scheduler, llm, trainable, losses
    torch.cuda.empty_cache()

    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        f"{PROJECT / 'src'}{os.pathsep}" + environment.get("PYTHONPATH", "")
    )
    output_csv = workspace / "ablation_generated_val.csv"
    subprocess.run(
        [
            sys.executable, "-m", "mrrate_report_training.generate",
            "--config", str(config_path),
            "--mode", "cached",
            "--split", "val",
            "--checkpoint", str(checkpoint_path),
            "--output-csv", str(output_csv),
            "--max-new-tokens", "48",
            "--overwrite",
        ],
        check=True,
        env=environment,
        cwd=PROJECT,
    )
    rows = list(csv.DictReader(output_csv.open()))
    assert [row["study_uid"] for row in rows] == SPLITS["val"]
    result = {
        "status": "PASS",
        "mil_conditioning": "none",
        "train_loss_one_step": train_loss,
        "trainable_tensors": len(state),
        "cross_mode_checkpoint_refused": cross_mode_refused,
        "studies_generated": len(rows),
        "sample_prediction": rows[0]["findings_pred"][:120],
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (workspace / "ablation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MRRATE_ABLATION_PROBE_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
