"""GPU training e2e: run the real train.py CLI in both conditioning modes.

Reuses the fabricated artifact workspace from tests/gpu_full_stack_e2e.py
(MRRATE_FULLSTACK_DIR), adds a ``train`` split to its exact token cache, and
then runs the actual trainer CLI end to end on real Gemma:

1. ``train.py --mode cached`` with ``mil_conditioning: all_classes``
   (frozen MIL head, provenance verified) for a few updates.
2. ``train.py --mode cached`` with ``mil_conditioning: none`` (the
   no-classification-labels ablation: no MIL head at all) for a few updates.
3. ``generate.py`` in ablation mode from the ablation training checkpoint.

Asserts the two checkpoints carry the right schema: the full one stores the
74 MIL label names and MIL tensors, the ablation one stores an empty label
schema, the ``none`` mode stamp, and no ``label_embeddings``/``mil_*``
tensors.
"""

from __future__ import annotations

import csv
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

from gpu_full_stack_e2e import (  # noqa: E402
    MIL_CLASSES,
    MIL_LABEL_NAMES,
    SPLITS,
    write_cache_split,
)

TRAINING_BLOCK = {
    "epochs": 1,
    "batch_size": 1,
    "gradient_accumulation": 1,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "warmup_ratio": 0.0,
    "checkpoint_every": 0,
    "shuffle": True,
    "replacement_sampling": False,
}
MAX_UPDATES = 4


def run_cli(module: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        f"{PROJECT / 'src'}{os.pathsep}" + environment.get("PYTHONPATH", "")
    )
    print(f"[train-e2e] CLI: {module} {' '.join(arguments[:4])} ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        check=True,
        env=environment,
        cwd=PROJECT,
    )


def write_training_config(
    base: dict, path: Path, *, mil_conditioning: str, output_dir: Path
) -> None:
    config = json.loads(json.dumps(base))
    config["writer"]["mil_conditioning"] = mil_conditioning
    config["training"] = dict(TRAINING_BLOCK)
    config["output_dir"] = str(output_dir)
    if mil_conditioning == "none":
        config["mil_checkpoint"] = "/placeholder/unused/mil_head.pt"
    path.write_text(yaml.safe_dump(config))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This e2e requires a CUDA GPU")
    workspace = Path(os.environ["MRRATE_FULLSTACK_DIR"])
    base_config = yaml.safe_load((workspace / "config.yaml").read_text())

    # The full-stack workspace has val/test caches; training needs a train
    # split. Reuse the same studies (their report targets already exist).
    cache_root = workspace / "exact_tokens"
    if not (cache_root / "token_features_train.json").exists():
        rng = np.random.default_rng(71)
        write_cache_split(
            cache_root, "train", SPLITS["val"] + SPLITS["test"], rng
        )
        print("[train-e2e] fabricated train cache split", flush=True)

    started = time.time()
    results = {}
    for mode in ("all_classes", "none"):
        config_path = workspace / f"config_train_{mode}.yaml"
        output_dir = workspace / f"train_runs_{mode}"
        write_training_config(
            base_config, config_path, mil_conditioning=mode, output_dir=output_dir
        )
        run_cli(
            "mrrate_report_training.train",
            "--config", str(config_path),
            "--mode", "cached",
            "--max-updates", str(MAX_UPDATES),
        )
        checkpoint = output_dir / "last.pt"
        package = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = package["trainable_state_dict"]
        has_mil_tensors = any(
            "label_embeddings" in name or "mil_" in name for name in state
        )
        stored_mode = package["config"]["writer"]["mil_conditioning"]
        if mode == "all_classes":
            assert package["label_names"] == MIL_LABEL_NAMES
            assert has_mil_tensors
            assert stored_mode == "all_classes"
        else:
            assert package["label_names"] == []
            assert not has_mil_tensors
            assert stored_mode == "none"
        lora_nonzero = sum(
            int(torch.count_nonzero(value) > 0)
            for name, value in state.items()
            if "lora_" in name
        )
        assert lora_nonzero > 0, f"{mode}: training left all LoRA tensors zero"
        results[mode] = {
            "updates": int(package["update"]),
            "trainable_tensors": len(state),
            "lora_tensors_nonzero": lora_nonzero,
            "checkpoint": str(checkpoint),
        }
        print(f"[train-e2e] {mode}: {results[mode]}", flush=True)

    # Generate from the ablation TRAINING checkpoint through the real CLI.
    generated_csv = workspace / "train_e2e_ablation_generated_val.csv"
    run_cli(
        "mrrate_report_training.generate",
        "--config", str(workspace / "config_train_none.yaml"),
        "--mode", "cached",
        "--split", "val",
        "--checkpoint", results["none"]["checkpoint"],
        "--output-csv", str(generated_csv),
        "--max-new-tokens", "48",
        "--overwrite",
    )
    rows = list(csv.DictReader(generated_csv.open()))
    assert [row["study_uid"] for row in rows] == SPLITS["val"]

    result = {
        "status": "PASS",
        "max_updates": MAX_UPDATES,
        "modes": results,
        "mil_classes_full": MIL_CLASSES,
        "ablation_generated_studies": len(rows),
        "elapsed_seconds": round(time.time() - started),
    }
    (workspace / "training_e2e_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("MRRATE_TRAINING_E2E_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
