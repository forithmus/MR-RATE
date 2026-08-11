"""One-stop evaluation of generated reports: NLG plus clinical accuracy.

Inputs are the generation CSV(s) from ``mrrate_report_training.generate``
(columns ``study_uid, findings_gt, findings_pred``), the ground-truth
pathology labels CSV, and the predicted labels CSV produced by
``mrrate_report_training.extract_labels`` on the generated reports. The
clinical block can be skipped (``--pred-labels`` omitted) to score NLG only,
e.g. while the GPU label-extraction job is still queued.

Outputs in --output-dir:
    metrics.json              corpus NLG metrics + clinical summary
    per_pathology_metrics.csv per-pathology confusion and metrics
    nlg_per_sample.csv        per-study NLG metrics

CLI:
    python -m mrrate_report_training.evaluate_reports \
        --generated-csv generated_test.csv \
        --gt-labels mrrate_labels.csv \
        --pred-labels pred_labels_test.csv \
        --output-dir runs/eval_test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .clinical_metrics import (
    _sanitize,
    align_labels,
    compute_clinical_metrics,
    write_metrics,
)
from .nlg_metrics import compute_nlg_metrics, load_generated_csv


def evaluate(
    generated_csvs: list[str],
    *,
    gt_labels: str | None,
    pred_labels: str | None,
    output_dir: str | Path,
    require_full_coverage: bool = False,
) -> dict:
    rows = load_generated_csv(generated_csvs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus, per_sample = compute_nlg_metrics(
        [row["findings_gt"] for row in rows],
        [row["findings_pred"] for row in rows],
    )
    fields = ["study_uid", *per_sample[0].keys()]
    with (output / "nlg_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, sample in zip(rows, per_sample):
            writer.writerow({"study_uid": row["study_uid"], **sample})

    clinical_summary = None
    if pred_labels:
        if not gt_labels:
            raise ValueError("--gt-labels is required with --pred-labels")
        pred_ids, names, gt_matrix, pred_matrix = align_labels(
            gt_labels,
            pred_labels,
            allow_missing_gt=not require_full_coverage,
        )
        generated_ids = {row["study_uid"] for row in rows}
        unmatched = sorted(generated_ids.symmetric_difference(pred_ids))
        if unmatched:
            raise ValueError(
                f"Generated reports and predicted labels cover different "
                f"studies; first={unmatched[:5]}"
            )
        clinical = compute_clinical_metrics(gt_matrix, pred_matrix, names)
        write_metrics(clinical, output)
        clinical_summary = clinical["summary"]

    metrics = _sanitize(
        {"nlg": corpus, "clinical": clinical_summary}
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-csv", nargs="+", required=True)
    parser.add_argument("--gt-labels")
    parser.add_argument("--pred-labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-full-coverage", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(
        args.generated_csv,
        gt_labels=args.gt_labels,
        pred_labels=args.pred_labels,
        output_dir=args.output_dir,
        require_full_coverage=args.require_full_coverage,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
