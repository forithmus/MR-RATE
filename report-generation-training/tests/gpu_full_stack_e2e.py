"""GPU full-stack trial: real Gemma-3-12B + PEFT LoRA with fabricated weights.

Unlike tests/e2e_dummy.py (duck-typed tiny LLM, CPU), this gate exercises the
production stack end to end with dummy *data* but real *code paths*:

1. Fabricates an 87-pathology schema, ground-truth reports/labels, a
   512-dim exact token cache (val + test), a dummy encoder checkpoint, and a
   74-class ``ClassifyThenAggregate`` MIL checkpoint whose recorded
   provenance (encoder SHA-256, architecture) satisfies preflight-grade
   verification.
2. Builds the real Gemma writer (``build_gemma_writer`` -> PEFT LoRA),
   runs one genuine optimizer step so the LoRA B matrices become nonzero,
   and saves a checkpoint through ``train.save_checkpoint`` — the exact
   trainer format.
3. Rebuilds a fresh Gemma writer (LoRA back at zero) and loads the
   checkpoint through ``load_writer_checkpoint``, asserting every trainable
   tensor — including every LoRA tensor — round-trips bit-exactly and that
   the nonzero LoRA state was restored.
4. Invokes the real ``mrrate_report_training.generate`` CLI (cached mode)
   for the val and test splits, then re-invokes with ``--resume`` to prove
   restart skips completed studies.
5. Runs keyword-backend label extraction and the evaluator over the
   generated reports and writes ``result.json``.

Run under Slurm via scripts/gpu_full_stack_e2e.sbatch.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from e2e_dummy import (  # noqa: E402
    NUM_PATHOLOGIES,
    build_ground_truth,
    pathology_names,
    resolve_upstream_root,
    write_pathologies_json,
)
from mrrate_report_training.cache import ExactRaggedTokenDataset  # noqa: E402
from mrrate_report_training.generate import load_writer_checkpoint  # noqa: E402
from mrrate_report_training.mil import infer_mil, load_frozen_mil  # noqa: E402
from mrrate_report_training.model import (  # noqa: E402
    ReportWriter,
    build_gemma_writer,
    label_semantic_embeddings,
    trainable_state_dict,
)
from mrrate_report_training.targets import load_target_index  # noqa: E402
from mrrate_report_training.train import cosine_schedule, save_checkpoint  # noqa: E402

DIM = 512
MIL_CLASSES = 74
MIL_LABEL_NAMES = [f"synthetic_mr_finding_{index:02d}" for index in range(MIL_CLASSES)]
GEMMA_PATH = os.environ.get(
    "MRRATE_LLM_PATH",
    "/hnvme/data/LLM-TZ/hub/models--google--gemma-3-12b-it/snapshots/"
    "96b6f1eccf38110c56df3a15bffe176da04bfd80",
)
SPLITS = {
    "val": [f"val_{index:02d}" for index in range(4)],
    "test": [f"test_{index:02d}" for index in range(4)],
}


def log(message: str) -> None:
    print(f"[full-stack] {message}", flush=True)


def write_cache_split(root: Path, split: str, subject_ids: list[str], rng) -> None:
    token_counts = [int(rng.integers(200, 400)) for _ in subject_ids]
    with (root / f"tokens_{split}.bin").open("wb") as handle:
        for count in token_counts:
            rng.normal(size=(count, DIM)).astype(np.float16).tofile(handle)
    offsets = np.concatenate(([0], np.cumsum(token_counts))).astype(np.int64)
    np.save(root / f"offsets_{split}.npy", offsets)
    np.save(
        root / f"labels_{split}.npy",
        rng.integers(0, 2, size=(len(subject_ids), MIL_CLASSES)).astype(np.float32),
    )
    np.save(
        root / f"full_counts_{split}.npy", np.asarray(token_counts, dtype=np.int64)
    )
    np.save(
        root / f"series_counts_{split}.npy",
        np.ones(len(subject_ids), dtype=np.int32),
    )
    (root / f"subject_ids_{split}.txt").write_text("\n".join(subject_ids) + "\n")
    (root / f"token_features_{split}.json").write_text(
        json.dumps(
            {
                "format": "raw_numpy_memmap",
                "feature_level": "projected_per_series_visual_tokens",
                "split": split,
                "tokens_file": f"tokens_{split}.bin",
                "offsets_file": f"offsets_{split}.npy",
                "labels_file": f"labels_{split}.npy",
                "subject_ids_file": f"subject_ids_{split}.txt",
                "full_token_counts_file": f"full_counts_{split}.npy",
                "series_counts_file": f"series_counts_{split}.npy",
                "dtype": "float16",
                "dim": DIM,
                "max_tokens_per_study": 0,
                "cache_fingerprint": "gpu_full_stack_dummy_v1",
            }
        )
    )


def fabricate_mil_checkpoint(
    workspace: Path, upstream: Path, encoder_config: dict
) -> tuple[Path, Path]:
    sys.path.insert(0, str(upstream / "scripts"))
    from mil_probe import ClassifyThenAggregate

    encoder_path = workspace / "encoder_dummy.pt"
    encoder_path.write_bytes(b"gpu full stack synthetic encoder identity v1")
    encoder_sha = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    torch.manual_seed(101)
    head = ClassifyThenAggregate(
        dim=DIM,
        n_classes=MIL_CLASSES,
        hidden_dim=512,
        mlp_hidden_dims=(384,),
        drop_rate=0.0,
        use_gating=True,
        use_norm=False,
        use_output_bias_scale=True,
    )
    mil_path = workspace / "mil_head_dummy.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "architecture": {
                "dim": DIM,
                "n_classes": MIL_CLASSES,
                "hidden_dim": 512,
                "mlp_hidden_dims": [384],
                "drop_rate": 0.0,
                "use_gating": True,
                "use_norm": False,
                "use_output_bias_scale": True,
            },
            "label_names": MIL_LABEL_NAMES,
            "validation_thresholds": [0.0] * MIL_CLASSES,
            "data_provenance": {
                "encoder_checkpoint": {"sha256": encoder_sha},
                "encoder": encoder_config,
            },
        },
        mil_path,
    )
    return mil_path, encoder_path


def build_writer(device: torch.device) -> ReportWriter:
    llm, tokenizer, _ = build_gemma_writer(
        GEMMA_PATH, device, lora_r=16, lora_alpha=32
    )
    semantics = label_semantic_embeddings(llm, tokenizer, MIL_LABEL_NAMES)
    return ReportWriter(
        llm,
        tokenizer,
        semantics,
        visual_dim=DIM,
        num_visual_queries=512,
        resampler_depth=2,
        resampler_heads=8,
        max_target_tokens=1536,
    ).to(device)


def lora_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if "lora_" in name}


def release(*objects) -> None:
    for value in objects:
        del value
    gc.collect()
    torch.cuda.empty_cache()


def run_cli(module: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        f"{PROJECT / 'src'}{os.pathsep}" + environment.get("PYTHONPATH", "")
    )
    log(f"CLI: {module} {' '.join(arguments[:6])} ...")
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        check=True,
        env=environment,
        cwd=PROJECT,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This full-stack gate requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    workspace = Path(
        os.environ.get("MRRATE_FULLSTACK_DIR", PROJECT / "runs" / f"gpu_full_stack_{job_id}")
    )
    workspace.mkdir(parents=True, exist_ok=True)
    log(f"workspace: {workspace}")
    rng = np.random.default_rng(53)
    upstream = resolve_upstream_root()

    # ---- Fabricated dataset, schema, cache, MIL/encoder artifacts ----------
    all_subjects = SPLITS["val"] + SPLITS["test"]
    findings, labels = build_ground_truth(rng, all_subjects)
    pathologies_json = workspace / "pathologies_dummy87.json"
    write_pathologies_json(pathologies_json)
    gt_labels_csv = workspace / "gt_labels.csv"
    with gt_labels_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_uid", *pathology_names()])
        for subject_id in all_subjects:
            writer.writerow([subject_id, *labels[subject_id].tolist()])
    reports_csv = workspace / "all_reports.csv"
    with reports_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_uid", "findings"])
        for subject_id in all_subjects:
            writer.writerow([subject_id, findings[subject_id]])
    cache_root = workspace / "exact_tokens"
    cache_root.mkdir(exist_ok=True)
    for split, subject_ids in SPLITS.items():
        write_cache_split(cache_root, split, subject_ids, rng)
    (cache_root / "label_names.json").write_text(json.dumps(MIL_LABEL_NAMES))

    encoder_config = {
        "name": "vjepa2",
        "vjepa21_checkpoint": None,
        "chunk_size": 64,
        "fusion_mode": "late",
        "pooling_strategy": "simple_attn",
        "extra_latent_projection": False,
        "dim_latent": DIM,
    }
    mil_path, encoder_path = fabricate_mil_checkpoint(
        workspace, upstream, encoder_config
    )
    config_path = workspace / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 17,
                "mode": "cached",
                "output_dir": str(workspace / "writer_runs"),
                "upstream_root": str(upstream),
                "encoder_checkpoint": str(encoder_path),
                "mil_checkpoint": str(mil_path),
                "llm_path": GEMMA_PATH,
                "data": {
                    "reports_csv": str(reports_csv),
                    "cached_tokens_dir": str(cache_root),
                },
                "encoder": encoder_config,
                "writer": {
                    "num_visual_queries": 512,
                    "resampler_depth": 2,
                    "resampler_heads": 8,
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "max_target_tokens": 1536,
                    "mil_conditioning": "all_classes",
                    "mil_proposal_dropout": 0.0,
                    "localization": False,
                },
            }
        )
    )

    # ---- Phase 1: real Gemma writer, one genuine LoRA update, save --------
    log("phase 1: loading Gemma-3-12B with report LoRA (train side)")
    started = time.time()
    mil_head, label_names, thresholds = load_frozen_mil(
        mil_path, upstream, expected_dim=DIM
    )
    assert label_names == MIL_LABEL_NAMES
    mil_head.to(device).eval()
    thresholds = thresholds.to(device)
    writer_model = build_writer(device)
    log(f"gemma loaded in {time.time() - started:.0f}s")

    targets = load_target_index(reports_csv)
    dataset = ExactRaggedTokenDataset(
        cache_root,
        "val",
        targets,
        expected_dim=DIM,
        expected_label_names=MIL_LABEL_NAMES,
    )
    zero_lora = sum(
        int(torch.count_nonzero(value) == 0)
        for value in lora_tensors(trainable_state_dict(writer_model)).values()
    )
    trainable = [value for value in writer_model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    scheduler = cosine_schedule(optimizer, total_updates=2, warmup_ratio=0.0)
    writer_model.train()
    item = dataset[0]
    tokens = item["tokens"].to(device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mil_logits = infer_mil(mil_head, tokens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = writer_model(tokens, mil_logits, thresholds, item["target"])
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    scheduler.step()
    train_loss = float(losses["loss"].detach())
    log(f"one training update done, loss={train_loss:.4f}")

    checkpoint_path = workspace / "last.pt"
    save_checkpoint(
        checkpoint_path,
        writer_model,
        optimizer,
        scheduler,
        {"synthetic_full_stack": True},
        label_names,
        epoch=0,
        next_slot=1,
        update=1,
        rank=0,
        world=1,
    )
    recorded = {
        name: value.clone() for name, value in trainable_state_dict(writer_model).items()
    }
    recorded_lora = lora_tensors(recorded)
    nonzero_lora_after_step = sum(
        int(torch.count_nonzero(value) > 0) for value in recorded_lora.values()
    )
    assert nonzero_lora_after_step > 0, "training step left every LoRA tensor zero"
    log(
        f"saved trainer-format checkpoint: {len(recorded)} trainable tensors, "
        f"{len(recorded_lora)} LoRA tensors "
        f"({nonzero_lora_after_step} nonzero after update, "
        f"{zero_lora} were zero at init)"
    )
    release(writer_model, optimizer, scheduler, trainable, losses)

    # ---- Phase 2: fresh Gemma writer, strict checkpoint load, LoRA proof --
    log("phase 2: fresh Gemma writer, loading checkpoint through inference path")
    fresh = build_writer(device)
    fresh_lora = lora_tensors(trainable_state_dict(fresh))
    assert set(fresh_lora) == set(recorded_lora)
    fresh_nonzero = sum(
        int(torch.count_nonzero(value) > 0) for value in fresh_lora.values()
    )
    package = load_writer_checkpoint(checkpoint_path, fresh, label_names)
    assert int(package["update"]) == 1
    loaded = trainable_state_dict(fresh)
    mismatched = [
        name
        for name, value in recorded.items()
        if not torch.equal(value, loaded[name])
    ]
    assert not mismatched, f"tensors changed in round trip: {mismatched[:5]}"
    loaded_nonzero = sum(
        int(torch.count_nonzero(value) > 0)
        for value in lora_tensors(loaded).values()
    )
    assert loaded_nonzero == nonzero_lora_after_step
    log(
        f"checkpoint round trip exact for all {len(recorded)} tensors; "
        f"LoRA nonzero: fresh={fresh_nonzero} -> loaded={loaded_nonzero}"
    )
    release(fresh, mil_head)

    # ---- Phase 3: the real generate.py CLI, cached val/test + resume ------
    generated = {}
    for split in SPLITS:
        output_csv = workspace / f"generated_{split}.csv"
        run_cli(
            "mrrate_report_training.generate",
            "--config", str(config_path),
            "--mode", "cached",
            "--split", split,
            "--checkpoint", str(checkpoint_path),
            "--output-csv", str(output_csv),
            "--max-new-tokens", "48",
        )
        rows = list(csv.DictReader(output_csv.open()))
        assert [row["study_uid"] for row in rows] == SPLITS[split]
        generated[split] = rows
        log(f"generated split={split}: {len(rows)} studies")
    # Resume must skip everything already generated and change nothing.
    before = (workspace / "generated_val.csv").read_text()
    run_cli(
        "mrrate_report_training.generate",
        "--config", str(config_path),
        "--mode", "cached",
        "--split", "val",
        "--checkpoint", str(checkpoint_path),
        "--output-csv", str(workspace / "generated_val.csv"),
        "--max-new-tokens", "48",
        "--resume",
    )
    assert (workspace / "generated_val.csv").read_text() == before
    log("resume run left completed output untouched")

    # ---- Phase 4: label extraction (keyword) + full evaluation ------------
    summaries = {}
    for split in SPLITS:
        pred_labels = workspace / f"pred_labels_{split}.csv"
        run_cli(
            "mrrate_report_training.extract_labels",
            "--generated-csv", str(workspace / f"generated_{split}.csv"),
            "--pathologies-json", str(pathologies_json),
            "--output-csv", str(pred_labels),
            "--backend", "keyword",
        )
        eval_dir = workspace / f"eval_{split}"
        run_cli(
            "mrrate_report_training.evaluate_reports",
            "--generated-csv", str(workspace / f"generated_{split}.csv"),
            "--gt-labels", str(gt_labels_csv),
            "--pred-labels", str(pred_labels),
            "--output-dir", str(eval_dir),
        )
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        assert metrics["clinical"]["pathologies"] == NUM_PATHOLOGIES
        summaries[split] = {
            "micro_f1": metrics["clinical"]["micro"]["f1"],
            "micro_sensitivity": metrics["clinical"]["micro"]["sensitivity"],
            "micro_specificity": metrics["clinical"]["micro"]["specificity"],
            "bleu4": metrics["nlg"]["bleu4"],
            "rougeL_f1": metrics["nlg"]["rougeL_f1"],
        }

    empty_predictions = sum(
        1
        for rows in generated.values()
        for row in rows
        if not row["findings_pred"].strip()
    )
    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "llm_path": GEMMA_PATH,
        "train_loss_one_step": train_loss,
        "trainable_tensors": len(recorded),
        "lora_tensors": len(recorded_lora),
        "lora_nonzero_after_step": nonzero_lora_after_step,
        "checkpoint_round_trip_exact": True,
        "resume_noop_verified": True,
        "studies_generated": sum(len(rows) for rows in generated.values()),
        "empty_predictions": empty_predictions,
        "pathologies": NUM_PATHOLOGIES,
        "mil_classes": MIL_CLASSES,
        "splits": summaries,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "workspace": str(workspace),
    }
    (workspace / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MRRATE_GPU_FULL_STACK_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
