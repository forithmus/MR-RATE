"""Extract NeuroVFM diagnosis labels from generated reports.

The ``vllm`` backend uses the same NeuroVFM MRI diagnosis extraction prompt,
74-diagnosis guidance schema, parser, and deterministic Gemma settings used to
build the current MR-RATE labels. Generated findings are staged as
``batch00_reports.csv`` and passed to
``data-preprocessing/.../07_neurovfm_diagnosis_extraction``.

The ``keyword`` backend is a deterministic name/synonym matcher with basic
negation handling. It exists so unit tests and the dummy end-to-end trial can
run without a GPU; it is NOT a clinically valid labeler.

Studies with an empty generated report receive all-zero labels (the extractor
silently drops empty findings rows; an empty report asserts no diagnosis).

CLI:
    python -m mrrate_report_training.extract_labels \
        --generated-csv generated_val.csv \
        --diagnoses-json neurovfm_mri_diagnoses.json \
        --output-csv pred_labels_val.csv \
        --backend vllm --work-dir runs/label_extraction_val
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    / "07_neurovfm_diagnosis_extraction"
)
_DEFAULT_DIAGNOSES_JSON = (
    _DEFAULT_CLASSIFIER_DIR / "data" / "neurovfm_mri_diagnoses.json"
)

_NEGATIONS = {"no", "not", "without", "absent", "negative", "denies", "denied"}
_NEGATION_WINDOW = 5


def load_pathology_schema(path: str | Path) -> dict[str, list[str]]:
    """Diagnosis key -> phrases used only by the test-only keyword backend.

    The current extraction schema is ``{"diagnoses": [{"key": ...}]}``. The
    older ``pathologies`` mapping and simple-list formats remain readable so
    existing CPU-only fixtures and historical artifacts can still be inspected.
    """

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "diagnoses" in data:
        diagnoses = data["diagnoses"]
        if not isinstance(diagnoses, list):
            raise ValueError(f"{path}: diagnoses must be a list")
        entries = {}
        for diagnosis in diagnoses:
            if not isinstance(diagnosis, dict) or not str(
                diagnosis.get("key", "")
            ).strip():
                raise ValueError(f"{path}: every diagnosis needs a non-empty key")
            key = str(diagnosis["key"])
            if key in entries:
                raise ValueError(f"{path}: duplicate diagnosis key {key!r}")
            entries[key] = diagnosis
    else:
        entries = data.get("pathologies", data) if isinstance(data, dict) else data
    if isinstance(entries, list):
        entries = {str(name): {} for name in entries}
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path}: no diagnoses found")
    schema: dict[str, list[str]] = {}
    for name, entry in entries.items():
        phrases = [str(name), str(name).replace("_", " ")]
        if isinstance(entry, dict):
            phrases.extend(str(value) for value in entry.get("synonyms", ()))
        schema[str(name)] = list(dict.fromkeys(phrases))
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


def _resume_labels_dir(
    rows: list[dict],
    *,
    work_dir: Path,
    text_column: str,
    diagnoses_json: Path,
    model_name: str | None,
    seed: int,
) -> Path:
    """Create or validate a resume directory without storing report text."""

    input_hash = hashlib.sha256()
    for row in rows:
        input_hash.update(str(row["study_uid"]).encode())
        input_hash.update(b"\0")
        input_hash.update(str(row[text_column] or "").encode())
        input_hash.update(b"\0")
    manifest = {
        "format": "mrrate_neurovfm_extraction_v1",
        "studies": len(rows),
        "input_sha256": input_hash.hexdigest(),
        "diagnoses_sha256": hashlib.sha256(diagnoses_json.read_bytes()).hexdigest(),
        "model": model_name or "google/gemma-4-31B-it",
        "seed": seed,
    }
    labels_dir = work_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "extraction_manifest.json"
    existing_shards = list(labels_dir.glob("labels_rank_*.json"))
    if existing_shards:
        if not manifest_path.exists():
            raise ValueError(
                f"{labels_dir} contains extraction shards without a manifest; "
                "use a new --work-dir"
            )
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise ValueError(
                f"{labels_dir} belongs to different reports, diagnoses, model, "
                "or seed; use a new --work-dir"
            )
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    return labels_dir


def extract_vllm_labels(
    rows: list[dict],
    schema: dict[str, list[str]],
    *,
    diagnoses_json: Path,
    work_dir: Path,
    classifier_dir: Path,
    text_column: str,
    model_name: str | None,
    batch_size: int,
    seed: int,
) -> list[dict]:
    classifier = classifier_dir / "extract_neurovfm_dx_gemma.py"
    merger = classifier_dir / "merge_labels.py"
    if not classifier.exists() or not merger.exists():
        raise FileNotFoundError(
            f"NeuroVFM diagnosis extractor not found under {classifier_dir}"
        )
    # Refuse the legacy 37-pathology schema: it belongs to the historical
    # three-step classifier and does not carry the NeuroVFM diagnosis guidance.
    payload = json.loads(diagnoses_json.read_text())
    diagnoses = payload.get("diagnoses") if isinstance(payload, dict) else None
    if not isinstance(diagnoses, list) or not diagnoses:
        raise ValueError(
            f"{diagnoses_json}: the vllm backend requires the NeuroVFM "
            'format {"diagnoses": [{"key": ..., "guidance": ...}, ...]}'
        )
    reports_dir, empty = _stage_reports(rows, work_dir, text_column)
    if len(empty) == len(rows):
        return [
            {"study_uid": row["study_uid"], "labels": {name: 0 for name in schema}}
            for row in rows
        ]
    # Keep rank files across requeues; the NeuroVFM extractor resumes globally
    # by study_uid and checkpoints atomically after every batch.
    labels_dir = _resume_labels_dir(
        rows,
        work_dir=work_dir,
        text_column=text_column,
        diagnoses_json=diagnoses_json,
        model_name=model_name,
        seed=seed,
    )
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
        "--diagnoses_json",
        str(diagnoses_json),
        "--output_dir",
        str(labels_dir),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
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
                "Classifier output schema differs from diagnoses JSON"
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
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--diagnoses-json",
        help="NeuroVFM diagnoses JSON; defaults to the bundled 74-diagnosis schema",
    )
    schema_group.add_argument(
        "--pathologies-json",
        dest="diagnoses_json",
        help="Deprecated alias retained for historical keyword-backend fixtures",
    )
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_generated_csv(args.generated_csv)
    if args.text_column not in rows[0]:
        raise ValueError(f"Unknown text column: {args.text_column}")
    diagnoses_json = Path(
        args.diagnoses_json or _DEFAULT_DIAGNOSES_JSON
    ).resolve()
    schema = load_pathology_schema(diagnoses_json)
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
            diagnoses_json=diagnoses_json,
            work_dir=Path(args.work_dir).resolve(),
            classifier_dir=Path(args.classifier_dir).resolve(),
            text_column=args.text_column,
            model_name=args.model_name,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    write_labels_csv(labeled, list(schema), args.output_csv)
    positives = sum(sum(row["labels"].values()) for row in labeled)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "studies": len(labeled),
                "diagnoses": len(schema),
                "positive_labels": positives,
                "output_csv": str(Path(args.output_csv).resolve()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
