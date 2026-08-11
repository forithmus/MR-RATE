"""Clinical accuracy metrics for generated reports.

Compares pathology labels extracted from generated reports against the
ground-truth pathology labels. The label schema is defined entirely by the
CSV headers (``study_uid`` followed by one binary column per pathology), so
the same code serves any schema size — including the 87-pathology set.

Per pathology: TP/FP/TN/FN, sensitivity (recall), specificity, precision
(PPV), NPV, F1, accuracy, balanced accuracy, prevalence, and support.
Aggregates: macro (NaN-safe mean over pathologies), micro (from pooled
confusion counts), weighted (positive-support weighted), and example-based
subset accuracy.

Undefined ratios (zero denominators, e.g. sensitivity of a pathology with no
positive ground-truth study) are reported as null and excluded from macro
averages.

CLI:
    python -m mrrate_report_training.clinical_metrics \
        --gt-labels gt.csv --pred-labels pred.csv --output-dir metrics/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_label_csv(path: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read a labels CSV into (study_uids, pathology_names, binary matrix)."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        if not fields:
            raise ValueError(f"{path}: empty CSV")
        id_column = "study_uid" if "study_uid" in fields else (
            "subject_id" if "subject_id" in fields else None
        )
        if id_column is None:
            raise ValueError(f"{path}: missing study_uid column")
        names = [field for field in fields if field != id_column]
        if not names:
            raise ValueError(f"{path}: no pathology columns")
        subject_ids: list[str] = []
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, 2):
            subject_id = str(row[id_column]).strip()
            if not subject_id:
                raise ValueError(f"{path}:{line_number}: empty {id_column}")
            values = []
            for name in names:
                raw = str(row[name]).strip()
                if raw not in ("0", "1", "0.0", "1.0"):
                    raise ValueError(
                        f"{path}:{line_number}: non-binary value {raw!r} "
                        f"for {name}"
                    )
                values.append(float(raw))
            subject_ids.append(subject_id)
            rows.append(values)
    if not subject_ids:
        raise ValueError(f"{path}: no label rows")
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(f"{path}: duplicate study_uid rows")
    return subject_ids, names, np.asarray(rows, dtype=np.float64)


def align_labels(
    gt_csv: str | Path,
    pred_csv: str | Path,
    *,
    allow_missing_gt: bool = True,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Align prediction rows to ground truth by study_uid and column name.

    Every predicted study must have ground truth. Ground-truth studies
    without predictions are allowed only when ``allow_missing_gt`` (the
    ground-truth CSV typically covers the full dataset, not one split).
    """

    gt_ids, gt_names, gt_matrix = load_label_csv(gt_csv)
    pred_ids, pred_names, pred_matrix = load_label_csv(pred_csv)
    if set(pred_names) != set(gt_names):
        missing = sorted(set(gt_names) - set(pred_names))[:5]
        extra = sorted(set(pred_names) - set(gt_names))[:5]
        raise ValueError(
            f"Label schemas differ: missing={missing} extra={extra}"
        )
    gt_index = {subject_id: row for subject_id, row in zip(gt_ids, gt_matrix)}
    unmatched = [value for value in pred_ids if value not in gt_index]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} predicted studies lack ground-truth labels; "
            f"first={unmatched[:5]}"
        )
    if not allow_missing_gt and len(pred_ids) != len(gt_ids):
        raise ValueError(
            f"Prediction rows ({len(pred_ids)}) do not cover ground truth "
            f"({len(gt_ids)})"
        )
    column_order = [pred_names.index(name) for name in gt_names]
    aligned_pred = pred_matrix[:, column_order]
    aligned_gt = np.stack([gt_index[value] for value in pred_ids])
    return pred_ids, gt_names, aligned_gt, aligned_pred


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _metric_block(tp: float, fp: float, tn: float, fn: float) -> dict:
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    balanced = (
        (sensitivity + specificity) / 2.0
        if not math.isnan(sensitivity) and not math.isnan(specificity)
        else math.nan
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": _ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": balanced,
    }


def compute_clinical_metrics(
    gt: np.ndarray, pred: np.ndarray, names: list[str]
) -> dict:
    if gt.shape != pred.shape or gt.shape[1] != len(names):
        raise ValueError("Ground truth, predictions, and names are misaligned")
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = (gt_bool & pred_bool).sum(axis=0).astype(float)
    fp = (~gt_bool & pred_bool).sum(axis=0).astype(float)
    fn = (gt_bool & ~pred_bool).sum(axis=0).astype(float)
    tn = (~gt_bool & ~pred_bool).sum(axis=0).astype(float)
    per_pathology = {}
    for index, name in enumerate(names):
        block = _metric_block(tp[index], fp[index], tn[index], fn[index])
        block["prevalence"] = float(gt_bool[:, index].mean())
        block["positives"] = int(gt_bool[:, index].sum())
        per_pathology[name] = block

    def macro(key: str) -> float | None:
        values = [
            per_pathology[name][key]
            for name in names
            if not math.isnan(per_pathology[name][key])
        ]
        return sum(values) / len(values) if values else None

    positives = np.array([per_pathology[name]["positives"] for name in names])
    weights = positives / positives.sum() if positives.sum() else None

    def weighted(key: str) -> float | None:
        if weights is None:
            return None
        total = 0.0
        for name, weight in zip(names, weights):
            if weight == 0.0:
                continue
            value = per_pathology[name][key]
            if math.isnan(value):
                return None
            total += weight * value
        return total

    metric_keys = (
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "f1",
        "accuracy",
        "balanced_accuracy",
    )
    micro = _metric_block(tp.sum(), fp.sum(), tn.sum(), fn.sum())
    summary = {
        "studies": int(gt.shape[0]),
        "pathologies": len(names),
        "subset_accuracy": float((gt_bool == pred_bool).all(axis=1).mean()),
        "hamming_accuracy": float((gt_bool == pred_bool).mean()),
        "macro": {key: macro(key) for key in metric_keys},
        "micro": {key: micro[key] for key in metric_keys},
        "weighted": {key: weighted(key) for key in metric_keys},
    }
    return {"summary": summary, "per_pathology": per_pathology}


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_metrics(result: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "clinical_metrics.json").write_text(
        json.dumps(_sanitize(result), indent=2, allow_nan=False) + "\n"
    )
    fields = ["pathology", *next(iter(result["per_pathology"].values()))]
    with (output / "per_pathology_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, block in result["per_pathology"].items():
            row = {
                key: "" if value is None else value
                for key, value in _sanitize(block).items()
            }
            writer.writerow({"pathology": name, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-labels", required=True)
    parser.add_argument("--pred-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail unless predictions cover every ground-truth study",
    )
    args = parser.parse_args()
    _, names, gt, pred = align_labels(
        args.gt_labels,
        args.pred_labels,
        allow_missing_gt=not args.require_full_coverage,
    )
    result = compute_clinical_metrics(gt, pred, names)
    write_metrics(result, args.output_dir)
    print(json.dumps(_sanitize(result["summary"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
