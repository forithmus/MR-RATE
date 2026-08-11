"""Overfit probe: train the real Gemma LoRA on 8 dummy studies, then check
that greedy generation reproduces each study's own findings.

Reuses the artifact workspace fabricated by tests/gpu_full_stack_e2e.py
(pass it via MRRATE_FULLSTACK_DIR). Trains ~N steps over the val+test
studies, saves a trainer-format checkpoint, reloads it through the
inference loader, and generates for every study. Success criterion:
generation is study-specific — the per-study token overlap with the study's
own ground truth is far higher than with other studies' ground truths.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from e2e_dummy import resolve_upstream_root  # noqa: E402
from gpu_full_stack_e2e import (  # noqa: E402
    DIM,
    MIL_LABEL_NAMES,
    SPLITS,
    build_writer,
)
from mrrate_report_training.cache import ExactRaggedTokenDataset  # noqa: E402
from mrrate_report_training.generate import load_writer_checkpoint  # noqa: E402
from mrrate_report_training.mil import infer_mil, load_frozen_mil  # noqa: E402
from mrrate_report_training.nlg_metrics import tokenize  # noqa: E402
from mrrate_report_training.targets import load_target_index  # noqa: E402
from mrrate_report_training.train import cosine_schedule, save_checkpoint  # noqa: E402

STEPS = int(os.environ.get("MRRATE_OVERFIT_STEPS", "800"))
LEARNING_RATE = float(os.environ.get("MRRATE_OVERFIT_LR", "3e-4"))


@torch.no_grad()
def teacher_forced_report(writer, tokens, mil_logits, thresholds, target):
    """Per-position argmax agreement of the training-style forward pass."""

    with torch.autocast("cuda", dtype=torch.bfloat16):
        visual_prefix, mil_tokens = writer.shared_prefix(
            tokens, mil_logits, thresholds
        )
        prefix = torch.cat((visual_prefix, mil_tokens), dim=1)
        prompt_ids = writer._token_ids(writer.REPORT_PROMPT, append_eos=False)
        target_ids = writer._token_ids(target.text, append_eos=True)
        embedding = writer.llm.get_input_embeddings()
        inputs = torch.cat(
            (
                prefix,
                embedding(prompt_ids).unsqueeze(0),
                embedding(target_ids).unsqueeze(0),
            ),
            dim=1,
        )
        attention_mask = torch.ones(
            1, inputs.shape[1], dtype=torch.long, device=inputs.device
        )
        token_type_ids = torch.zeros_like(attention_mask)
        token_type_ids[:, : prefix.shape[1]] = 1
        outputs = writer.llm(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=False,
            logits_to_keep=target_ids.numel() + 1,
        )
    logits = outputs.logits[:, :-1].float()[0]
    losses = torch.nn.functional.cross_entropy(
        logits, target_ids, reduction="none"
    )
    predicted = logits.argmax(-1)
    matches = predicted == target_ids
    worst = losses.argsort(descending=True)[:3]
    return {
        "argmax_accuracy": float(matches.float().mean()),
        "mismatches": int((~matches).sum()),
        "worst_positions": [
            {
                "position": int(index),
                "target": writer.tokenizer.decode([int(target_ids[index])]),
                "predicted": writer.tokenizer.decode([int(predicted[index])]),
                "loss": round(float(losses[index]), 3),
            }
            for index in worst
        ],
    }


def token_f1(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    return 2.0 * overlap / (len(a) + len(b))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    workspace = Path(os.environ["MRRATE_FULLSTACK_DIR"])
    upstream = resolve_upstream_root()

    mil_head, label_names, thresholds = load_frozen_mil(
        workspace / "mil_head_dummy.pt", upstream, expected_dim=DIM
    )
    assert label_names == MIL_LABEL_NAMES
    mil_head.to(device).eval()
    thresholds = thresholds.to(device)
    targets = load_target_index(workspace / "all_reports.csv")
    items = []
    for split in SPLITS:
        dataset = ExactRaggedTokenDataset(
            workspace / "exact_tokens",
            split,
            targets,
            expected_dim=DIM,
            expected_label_names=MIL_LABEL_NAMES,
        )
        for index in range(len(dataset)):
            item = dataset[index]
            tokens = item["tokens"].to(device=device, dtype=torch.bfloat16)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = infer_mil(mil_head, tokens)
            items.append((item["subject_id"], tokens, logits, item["target"]))
    print(f"[overfit] {len(items)} studies, {STEPS} steps, lr={LEARNING_RATE}", flush=True)

    writer = build_writer(device)
    trainable = [value for value in writer.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = cosine_schedule(optimizer, total_updates=STEPS, warmup_ratio=0.05)
    writer.train()
    started = time.time()
    for step in range(STEPS):
        subject_id, tokens, logits, target = items[step % len(items)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = writer(tokens, logits, thresholds, target)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(
                f"[overfit] step={step + 1}/{STEPS} "
                f"loss={float(losses['loss'].detach()):.4f} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    checkpoint_path = workspace / "overfit.pt"
    save_checkpoint(
        checkpoint_path, writer, optimizer, scheduler, {"overfit_probe": True},
        label_names, epoch=0, next_slot=0, update=STEPS, rank=0, world=1,
    )
    # Reload through the inference path into the same architecture.
    del optimizer, scheduler
    writer.eval()
    load_writer_checkpoint(checkpoint_path, writer, label_names)

    rows = []
    teacher_forced = {}
    cache_agreements = 0
    for subject_id, tokens, logits, target in items:
        teacher_forced[subject_id] = teacher_forced_report(
            writer, tokens, logits, thresholds, target
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = writer.generate(
                tokens, logits, thresholds, max_new_tokens=96
            )
            uncached = writer.generate(
                tokens, logits, thresholds, max_new_tokens=96, use_cache=False
            )
        cache_agreements += int(prediction == uncached)
        rows.append((subject_id, target.text, prediction))
        print(f"--- {subject_id} ---", flush=True)
        print("GT:  " + target.text.replace("\n", " | "), flush=True)
        print("PRED:" + " " + prediction.replace("\n", " | "), flush=True)
        if prediction != uncached:
            print("PRED(no-cache) DIFFERS: " + uncached.replace("\n", " | "), flush=True)
        print(
            "teacher-forced: "
            + json.dumps(teacher_forced[subject_id]["worst_positions"]),
            flush=True,
        )

    with (workspace / "overfit_generated.csv").open("w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["study_uid", "findings_gt", "findings_pred"])
        csv_writer.writerows(rows)

    # Study-specificity: own-GT overlap must beat cross-GT overlap.
    own = [token_f1(gt, pred) for _, gt, pred in rows]
    cross = [
        max(
            token_f1(other_gt, pred)
            for other_id, other_gt, _ in rows
            if other_id != subject_id
        )
        for subject_id, _, pred in rows
    ]
    specific = sum(int(o > c) for o, c in zip(own, cross))
    result = {
        "status": "PASS" if specific >= len(rows) - 1 else "WEAK",
        "steps": STEPS,
        "mean_own_token_f1": sum(own) / len(own),
        "mean_best_cross_token_f1": sum(cross) / len(cross),
        "studies_more_similar_to_own_gt": f"{specific}/{len(rows)}",
        "cached_uncached_generation_identical": f"{cache_agreements}/{len(rows)}",
        "per_study_own_f1": {row[0]: round(f, 3) for row, f in zip(rows, own)},
        "teacher_forced_argmax_accuracy": {
            subject_id: round(report["argmax_accuracy"], 4)
            for subject_id, report in teacher_forced.items()
        },
    }
    (workspace / "overfit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MRRATE_OVERFIT_PROBE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
