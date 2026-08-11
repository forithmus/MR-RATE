"""End-to-end dummy trial: generate -> extract labels -> clinical + NLG metrics.

Runs on CPU or GPU without external artifacts. It fabricates:

- an 87-pathology schema JSON (mirroring the extracted pathology set size),
- ground-truth findings whose pathology mentions define exact GT labels,
- an exact ragged token cache with ``val`` and ``test`` splits,
- a frozen 5-class ClassifyThenAggregate MIL head from the in-repo upstream,
- a tiny word-level writer LLM (duck-typed like tests/gpu_e2e.py).

It then exercises the real pipeline: checkpoint save/load through
``load_writer_checkpoint``, greedy ``ReportWriter.generate`` over both
splits, the ``extract_labels`` CLI (keyword backend), and the
``evaluate_reports`` CLI. A control evaluation labels the ground-truth text
itself and must reach perfect clinical agreement.

Run:  python tests/e2e_dummy.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mrrate_report_training.cache import ExactRaggedTokenDataset  # noqa: E402
from mrrate_report_training.generate import load_writer_checkpoint  # noqa: E402
from mrrate_report_training.mil import infer_mil  # noqa: E402
from mrrate_report_training.model import (  # noqa: E402
    ReportWriter,
    trainable_state_dict,
)
from mrrate_report_training.targets import load_target_index  # noqa: E402

NUM_PATHOLOGIES = 87
MIL_CLASSES = 5
DIM = 64
MIL_LABEL_NAMES = [f"mil_label_{index}" for index in range(MIL_CLASSES)]


def resolve_upstream_root() -> Path:
    candidates = [
        os.environ.get("MRRATE_UPSTREAM_ROOT"),
        PROJECT.parent / "contrastive-pretraining",
        "/hnvme/workspace/b180dc51-sezgin/MR-RATE-linearprobe/contrastive-pretraining",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "scripts" / "mil_probe.py").exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No upstream contrastive-pretraining checkout with scripts/mil_probe.py"
    )


def pathology_names() -> list[str]:
    return [f"Dummy pathology finding{index:02d}" for index in range(NUM_PATHOLOGIES)]


def write_pathologies_json(path: Path) -> None:
    payload = {
        "pathologies": {
            name: {
                "positive": f"There is {name.split()[-1]}",
                "negative": f"There is no {name.split()[-1]}",
                "synonyms": [name.split()[-1]],
            }
            for name in pathology_names()
        }
    }
    path.write_text(json.dumps(payload, indent=2))


def build_ground_truth(rng: np.random.Generator, subject_ids: list[str]):
    """Findings text with mention/negation patterns and the implied labels."""

    names = pathology_names()
    labels = {}
    findings = {}
    for subject_id in subject_ids:
        positive = rng.choice(NUM_PATHOLOGIES, size=4, replace=False)
        negated = [index for index in rng.choice(NUM_PATHOLOGIES, 3, replace=False)
                   if index not in positive]
        sentences = [
            f"There is {names[index].split()[-1]}." for index in sorted(positive)
        ] + [
            f"There is no {names[index].split()[-1]}." for index in sorted(negated)
        ]
        findings[subject_id] = "\n".join(sentences)
        row = np.zeros(NUM_PATHOLOGIES, dtype=np.int64)
        row[list(positive)] = 1
        labels[subject_id] = row
    return findings, labels


def write_cache_split(root: Path, split: str, subject_ids: list[str], rng) -> None:
    token_counts = [int(rng.integers(40, 90)) for _ in subject_ids]
    bags = [
        rng.normal(size=(count, DIM)).astype(np.float16) for count in token_counts
    ]
    with (root / f"tokens_{split}.bin").open("wb") as handle:
        for bag in bags:
            bag.tofile(handle)
    offsets = np.concatenate(([0], np.cumsum(token_counts))).astype(np.int64)
    np.save(root / f"offsets_{split}.npy", offsets)
    np.save(
        root / f"labels_{split}.npy",
        rng.integers(0, 2, size=(len(bags), MIL_CLASSES)).astype(np.float32),
    )
    np.save(
        root / f"full_counts_{split}.npy", np.asarray(token_counts, dtype=np.int64)
    )
    np.save(root / f"series_counts_{split}.npy", np.ones(len(bags), dtype=np.int32))
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
            }
        )
    )


class WordTokenizer:
    """Word-level tokenizer whose decode emits real report words."""

    def __init__(self) -> None:
        words = ["<pad>", "<eos>", "there", "is", "no", "acute", "chronic", "."]
        words += [f"finding{index:02d}" for index in range(NUM_PATHOLOGIES)]
        self.words = words
        self.index = {word: position for position, word in enumerate(words)}
        self.eos_token_id = self.index["<eos>"]

    def __call__(self, text, **kwargs):
        maximum = int(kwargs.get("max_length", 64))
        values = [
            self.index.get(token.lower().strip("."), self.index["."])
            for token in str(text).split()
        ][:maximum] or [self.index["."]]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, ids, **_):
        return " ".join(
            self.words[int(value)]
            for value in ids
            if int(value) not in (0, self.eos_token_id)
        )


class TinyReportAdapterLLM(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        vocabulary = len(WordTokenizer().words)
        torch.manual_seed(23)
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.output = nn.Linear(hidden, vocabulary)
        self.lora_A_report = nn.Parameter(torch.randn(hidden, 4) * 0.02)
        self.lora_B_report = nn.Parameter(torch.zeros(4, hidden))
        self.embedding.requires_grad_(False)
        self.output.requires_grad_(False)

    def set_adapter(self, name: str) -> None:
        if str(name) != "report":
            raise ValueError(f"unexpected adapter: {name}")

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, logits_to_keep, **_):
        hidden = inputs_embeds[:, -int(logits_to_keep):]
        hidden = hidden + hidden @ self.lora_A_report @ self.lora_B_report
        return SimpleNamespace(logits=self.output(hidden))


def build_writer(device: torch.device) -> ReportWriter:
    torch.manual_seed(29)
    return ReportWriter(
        TinyReportAdapterLLM(),
        WordTokenizer(),
        torch.randn(MIL_CLASSES, 32),
        visual_dim=DIM,
        num_visual_queries=8,
        resampler_depth=1,
        resampler_heads=4,
        max_target_tokens=48,
    ).to(device)


def run_cli(module: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        f"{PROJECT / 'src'}{os.pathsep}" + environment.get("PYTHONPATH", "")
    )
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        check=True,
        env=environment,
        cwd=PROJECT,
    )


def main() -> None:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    rng = np.random.default_rng(31)
    output = Path(
        os.environ.get(
            "MRRATE_E2E_DIR", tempfile.mkdtemp(prefix="mrrate_e2e_dummy_")
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    print(f"[e2e] workspace: {output}", flush=True)

    splits = {
        "val": [f"val_{index:02d}" for index in range(6)],
        "test": [f"test_{index:02d}" for index in range(8)],
    }
    all_subjects = splits["val"] + splits["test"]
    findings, labels = build_ground_truth(rng, all_subjects)

    pathologies_json = output / "pathologies_dummy87.json"
    write_pathologies_json(pathologies_json)
    gt_labels_csv = output / "gt_labels.csv"
    with gt_labels_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_uid", *pathology_names()])
        for subject_id in all_subjects:
            writer.writerow([subject_id, *labels[subject_id].tolist()])
    reports_csv = output / "all_reports.csv"
    with reports_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_uid", "findings"])
        for subject_id in all_subjects:
            writer.writerow([subject_id, findings[subject_id]])

    cache_root = output / "exact_tokens"
    cache_root.mkdir(exist_ok=True)
    for split, subject_ids in splits.items():
        write_cache_split(cache_root, split, subject_ids, rng)
    (cache_root / "label_names.json").write_text(json.dumps(MIL_LABEL_NAMES))

    sys.path.insert(0, str(resolve_upstream_root() / "scripts"))
    from mil_probe import ClassifyThenAggregate

    torch.manual_seed(37)
    mil = ClassifyThenAggregate(
        dim=DIM, n_classes=MIL_CLASSES, hidden_dim=16, mlp_hidden_dims=(12,)
    ).to(device)
    mil.requires_grad_(False)
    mil.eval()
    thresholds = torch.full((MIL_CLASSES,), 0.5, device=device)

    # Checkpoint round trip through the real inference loading path.
    trained = build_writer(device)
    with torch.no_grad():
        trained.llm.lora_B_report.add_(0.01)
    label_names = MIL_LABEL_NAMES
    checkpoint_path = output / "last.pt"
    torch.save(
        {
            "trainable_state_dict": trainable_state_dict(trained),
            "label_names": label_names,
            "update": 42,
        },
        checkpoint_path,
    )
    writer_model = build_writer(device)
    load_writer_checkpoint(checkpoint_path, writer_model, label_names)
    torch.testing.assert_close(
        writer_model.llm.lora_B_report, trained.llm.lora_B_report
    )
    writer_model.eval()

    targets = load_target_index(reports_csv)
    generated_csvs = []
    for split, subject_ids in splits.items():
        dataset = ExactRaggedTokenDataset(
            cache_root,
            split,
            targets,
            expected_dim=DIM,
            expected_label_names=MIL_LABEL_NAMES,
        )
        assert dataset.subject_ids == subject_ids
        generated_csv = output / f"generated_{split}.csv"
        with generated_csv.open("w", newline="") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["study_uid", "findings_gt", "findings_pred"])
            for index in range(len(dataset)):
                item = dataset[index]
                tokens = item["tokens"].to(device=device, dtype=torch.float32)
                mil_logits = infer_mil(mil, tokens)
                prediction = writer_model.generate(
                    tokens, mil_logits, thresholds, max_new_tokens=32
                )
                assert prediction, "generation must produce non-empty text"
                csv_writer.writerow(
                    [item["subject_id"], targets[item["subject_id"]].text, prediction]
                )
        generated_csvs.append(generated_csv)
        print(f"[e2e] generated split={split} studies={len(dataset)}", flush=True)

    # Control: labels extracted from the ground-truth text must match GT.
    control_labels_csv = output / "control_labels.csv"
    run_cli(
        "mrrate_report_training.extract_labels",
        "--generated-csv",
        *map(str, generated_csvs),
        "--pathologies-json",
        str(pathologies_json),
        "--output-csv",
        str(control_labels_csv),
        "--backend",
        "keyword",
        "--text-column",
        "findings_gt",
    )
    control_dir = output / "eval_control"
    run_cli(
        "mrrate_report_training.evaluate_reports",
        "--generated-csv",
        *map(str, generated_csvs),
        "--gt-labels",
        str(gt_labels_csv),
        "--pred-labels",
        str(control_labels_csv),
        "--output-dir",
        str(control_dir),
    )
    control = json.loads((control_dir / "metrics.json").read_text())
    assert control["clinical"]["subset_accuracy"] == 1.0, control["clinical"]
    assert control["clinical"]["micro"]["f1"] == 1.0

    # Real path: label the generated reports, evaluate per split.
    summaries = {}
    for split in splits:
        pred_labels_csv = output / f"pred_labels_{split}.csv"
        run_cli(
            "mrrate_report_training.extract_labels",
            "--generated-csv",
            str(output / f"generated_{split}.csv"),
            "--pathologies-json",
            str(pathologies_json),
            "--output-csv",
            str(pred_labels_csv),
            "--backend",
            "keyword",
        )
        eval_dir = output / f"eval_{split}"
        run_cli(
            "mrrate_report_training.evaluate_reports",
            "--generated-csv",
            str(output / f"generated_{split}.csv"),
            "--gt-labels",
            str(gt_labels_csv),
            "--pred-labels",
            str(pred_labels_csv),
            "--output-dir",
            str(eval_dir),
        )
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        assert metrics["nlg"]["samples"] == len(splits[split])
        assert 0.0 <= metrics["nlg"]["bleu1"] <= 1.0
        assert metrics["clinical"]["pathologies"] == NUM_PATHOLOGIES
        per_pathology = list(
            csv.DictReader((eval_dir / "per_pathology_metrics.csv").open())
        )
        assert len(per_pathology) == NUM_PATHOLOGIES
        assert (eval_dir / "nlg_per_sample.csv").exists()
        summaries[split] = {
            "micro_f1": metrics["clinical"]["micro"]["f1"],
            "micro_sensitivity": metrics["clinical"]["micro"]["sensitivity"],
            "micro_specificity": metrics["clinical"]["micro"]["specificity"],
            "bleu4": metrics["nlg"]["bleu4"],
            "rougeL_f1": metrics["nlg"]["rougeL_f1"],
        }

    # Ablation: a mil_conditioning=none writer must run the same inference
    # and produce the same per-class clinical evaluation over all pathologies.
    torch.manual_seed(29)
    ablation_writer = ReportWriter(
        TinyReportAdapterLLM(),
        WordTokenizer(),
        None,
        visual_dim=DIM,
        num_visual_queries=8,
        resampler_depth=1,
        resampler_heads=4,
        max_target_tokens=48,
        mil_conditioning="none",
        llm_dim=32,
    ).to(device)
    ablation_checkpoint = output / "ablation_last.pt"
    torch.save(
        {
            "trainable_state_dict": trainable_state_dict(ablation_writer),
            "label_names": [],
            "config": {"writer": {"mil_conditioning": "none"}},
            "update": 1,
        },
        ablation_checkpoint,
    )
    ablation_loaded = ReportWriter(
        TinyReportAdapterLLM(),
        WordTokenizer(),
        None,
        visual_dim=DIM,
        num_visual_queries=8,
        resampler_depth=1,
        resampler_heads=4,
        max_target_tokens=48,
        mil_conditioning="none",
        llm_dim=32,
    ).to(device)
    load_writer_checkpoint(ablation_checkpoint, ablation_loaded, [])
    ablation_loaded.eval()
    ablation_dataset = ExactRaggedTokenDataset(
        cache_root,
        "val",
        targets,
        expected_dim=DIM,
    )
    ablation_csv = output / "ablation_generated_val.csv"
    with ablation_csv.open("w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["study_uid", "findings_gt", "findings_pred"])
        for index in range(len(ablation_dataset)):
            item = ablation_dataset[index]
            tokens = item["tokens"].to(device=device, dtype=torch.float32)
            prediction = ablation_loaded.generate(
                tokens, None, None, max_new_tokens=32
            )
            csv_writer.writerow(
                [item["subject_id"], targets[item["subject_id"]].text, prediction]
            )
    ablation_pred_labels = output / "ablation_pred_labels_val.csv"
    run_cli(
        "mrrate_report_training.extract_labels",
        "--generated-csv",
        str(ablation_csv),
        "--pathologies-json",
        str(pathologies_json),
        "--output-csv",
        str(ablation_pred_labels),
        "--backend",
        "keyword",
    )
    ablation_eval = output / "eval_ablation_val"
    run_cli(
        "mrrate_report_training.evaluate_reports",
        "--generated-csv",
        str(ablation_csv),
        "--gt-labels",
        str(gt_labels_csv),
        "--pred-labels",
        str(ablation_pred_labels),
        "--output-dir",
        str(ablation_eval),
    )
    ablation_metrics = json.loads((ablation_eval / "metrics.json").read_text())
    assert ablation_metrics["clinical"]["pathologies"] == NUM_PATHOLOGIES
    assert (
        len(list(csv.DictReader((ablation_eval / "per_pathology_metrics.csv").open())))
        == NUM_PATHOLOGIES
    )
    print(
        f"[e2e] ablation (mil_conditioning=none) evaluated over "
        f"{NUM_PATHOLOGIES} classes",
        flush=True,
    )

    result = {
        "status": "PASS",
        "device": str(device),
        "workspace": str(output),
        "pathologies": NUM_PATHOLOGIES,
        "control_subset_accuracy": control["clinical"]["subset_accuracy"],
        "ablation_classes_evaluated": ablation_metrics["clinical"]["pathologies"],
        "splits": summaries,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MRRATE_E2E_DUMMY_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
