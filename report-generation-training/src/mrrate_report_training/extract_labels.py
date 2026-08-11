"""Extract pathology labels from generated reports.

The ``vllm`` backend reuses the repository's LLM pathology-classification
pipeline (``data-preprocessing/.../06_pathology_classification``) unchanged:
generated findings are staged as ``batch00_reports.csv``, the three-step
classifier runs on a GPU node, and ``merge_labels.py`` produces a labels CSV
with the same schema as the ground-truth labels — one binary column per
pathology in the supplied pathologies JSON (any schema size, e.g. 87).

The ``keyword`` backend is a deterministic name/synonym matcher with basic
negation handling. It exists so unit tests and the dummy end-to-end trial can
run without a GPU; it is NOT a clinically valid labeler.

Studies with an empty generated report receive all-zero labels (the upstream
classifier silently drops empty findings rows; an empty report asserts no
pathology).

CLI:
    python -m mrrate_report_training.extract_labels \
        --generated-csv generated_val.csv \
        --pathologies-json pathologies.json \
        --output-csv pred_labels_val.csv \
        --backend vllm --work-dir runs/label_extraction_val
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .nlg_metrics import load_generated_csv


_DEFAULT_CLASSIFIER_DIR = (
    Path(__file__).resolve().parents[3]
    / "data-preprocessing"
    / "src"
    / "mr_rate_preprocessing"
    / "reports_preprocessing"
    / "06_pathology_classification"
)

_NEGATIONS = {"no", "not", "without", "absent", "negative", "denies", "denied"}
_NEGATION_WINDOW = 5


def load_pathology_schema(path: str | Path) -> dict[str, list[str]]:
    """Pathology name -> match phrases (name plus optional 'synonyms')."""

    data = json.loads(Path(path).read_text())
    entries = data.get("pathologies", data) if isinstance(data, dict) else data
    if isinstance(entries, list):
        entries = {str(name): {} for name in entries}
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path}: no pathologies found")
    schema: dict[str, list[str]] = {}
    for name, entry in entries.items():
        phrases = [str(name)]
        if isinstance(entry, dict):
            phrases.extend(str(value) for value in entry.get("synonyms", ()))
        schema[str(name)] = phrases
    return schema


def _keyword_label(text: str, phrases: list[str]) -> int:
    # Negation never crosses a sentence boundary.
    sentences = [
        re.findall(r"[a-z0-9]+", sentence.lower())
        for sentence in re.split(r"[.;:!?\n]", str(text or ""))
    ]
    for phrase in phrases:
        phrase_tokens = re.findall(r"[a-z0-9]+", phrase.lower())
        if not phrase_tokens:
            continue
        span = len(phrase_tokens)
        for tokens in sentences:
            for start in range(len(tokens) - span + 1):
                if tokens[start : start + span] != phrase_tokens:
                    continue
                window = tokens[max(0, start - _NEGATION_WINDOW) : start]
                if not _NEGATIONS.intersection(window):
                    return 1
    return 0


def extract_keyword_labels(
    rows: list[dict], schema: dict[str, list[str]], *, text_column: str
) -> list[dict]:
    labeled = []
    for row in rows:
        text = row[text_column]
        labeled.append(
            {
                "study_uid": row["study_uid"],
                "labels": {
                    name: _keyword_label(text, phrases)
                    for name, phrases in schema.items()
                },
            }
        )
    return labeled


def write_labels_csv(
    labeled: list[dict], names: list[str], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_uid", *names])
        for row in sorted(labeled, key=lambda value: value["study_uid"]):
            writer.writerow(
                [row["study_uid"], *(row["labels"][name] for name in names)]
            )


def _fresh_directory(path: Path) -> Path:
    """Recreate a staging directory so stale outputs cannot be merged."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _stage_reports(rows: list[dict], work_dir: Path, text_column: str) -> tuple[Path, list[str]]:
    """Write non-empty findings as batch00_reports.csv; return empty uids."""

    reports_dir = _fresh_directory(work_dir / "reports")
    empty: list[str] = []
    with (reports_dir / "batch00_reports.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_uid", "findings"])
        for row in rows:
            text = str(row[text_column] or "").strip()
            if text:
                writer.writerow([row["study_uid"], text])
            else:
                empty.append(row["study_uid"])
    return reports_dir, empty


def extract_vllm_labels(
    rows: list[dict],
    schema: dict[str, list[str]],
    *,
    pathologies_json: Path,
    work_dir: Path,
    classifier_dir: Path,
    text_column: str,
    model_name: str | None,
    batch_size: int,
    seed: int,
    max_retries: int,
) -> list[dict]:
    classifier = classifier_dir / "classify_pathologies_parallel.py"
    merger = classifier_dir / "merge_labels.py"
    if not classifier.exists() or not merger.exists():
        raise FileNotFoundError(
            f"Pathology classifier not found under {classifier_dir}"
        )
    # The upstream classifier requires the structured pathologies format.
    payload = json.loads(pathologies_json.read_text())
    if not isinstance(payload, dict) or not isinstance(
        payload.get("pathologies"), dict
    ):
        raise ValueError(
            f"{pathologies_json}: the vllm backend requires the structured "
            'format {"pathologies": {name: {...}}}'
        )
    reports_dir, empty = _stage_reports(rows, work_dir, text_column)
    if len(empty) == len(rows):
        return [
            {"study_uid": row["study_uid"], "labels": {name: 0 for name in schema}}
            for row in rows
        ]
    labels_dir = _fresh_directory(work_dir / "labels")
    # The classifier shards by SLURM_PROCID/SLURM_NTASKS at import time; this
    # single subprocess must always see the whole staged CSV.
    environment = {
        **os.environ,
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": os.environ.get("SLURM_LOCALID", "0"),
    }
    command = [
        sys.executable,
        str(classifier),
        "--reports_dir",
        str(reports_dir),
        "--pathologies_json",
        str(pathologies_json),
        "--output_dir",
        str(labels_dir),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--max_retries",
        str(max_retries),
    ]
    if model_name:
        command.extend(["--model_name", model_name])
    subprocess.run(command, check=True, env=environment)
    merged_csv = work_dir / "merged_labels.csv"
    subprocess.run(
        [
            sys.executable,
            str(merger),
            "--input_dir",
            str(labels_dir),
            "--output",
            str(merged_csv),
        ],
        check=True,
        env=environment,
    )
    with merged_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = [field for field in reader.fieldnames or () if field != "study_uid"]
        if set(names) != set(schema):
            raise ValueError(
                "Classifier output schema differs from pathologies JSON"
            )
        labeled = [
            {
                "study_uid": str(row["study_uid"]),
                "labels": {name: int(float(row[name])) for name in names},
            }
            for row in reader
        ]
    labeled.extend(
        {"study_uid": subject_id, "labels": {name: 0 for name in schema}}
        for subject_id in empty
    )
    expected = {row["study_uid"] for row in rows}
    produced = {row["study_uid"] for row in labeled}
    if produced != expected:
        missing = sorted(expected - produced)[:5]
        raise ValueError(f"Classifier lost studies; first missing={missing}")
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-csv", nargs="+", required=True)
    parser.add_argument("--pathologies-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--backend", choices=("vllm", "keyword"), default="vllm")
    parser.add_argument(
        "--text-column",
        default="findings_pred",
        help="Column to label (findings_pred, or findings_gt for an "
        "extraction upper bound)",
    )
    parser.add_argument("--work-dir", help="Required for the vllm backend")
    parser.add_argument(
        "--classifier-dir", default=str(_DEFAULT_CLASSIFIER_DIR)
    )
    parser.add_argument("--model-name")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    rows = load_generated_csv(args.generated_csv)
    if args.text_column not in rows[0]:
        raise ValueError(f"Unknown text column: {args.text_column}")
    schema = load_pathology_schema(args.pathologies_json)
    if args.backend == "keyword":
        labeled = extract_keyword_labels(
            rows, schema, text_column=args.text_column
        )
    else:
        if not args.work_dir:
            raise ValueError("--work-dir is required for the vllm backend")
        labeled = extract_vllm_labels(
            rows,
            schema,
            pathologies_json=Path(args.pathologies_json).resolve(),
            work_dir=Path(args.work_dir).resolve(),
            classifier_dir=Path(args.classifier_dir).resolve(),
            text_column=args.text_column,
            model_name=args.model_name,
            batch_size=args.batch_size,
            seed=args.seed,
            max_retries=args.max_retries,
        )
    write_labels_csv(labeled, list(schema), args.output_csv)
    positives = sum(sum(row["labels"].values()) for row in labeled)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "studies": len(labeled),
                "pathologies": len(schema),
                "positive_labels": positives,
                "output_csv": str(Path(args.output_csv).resolve()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
