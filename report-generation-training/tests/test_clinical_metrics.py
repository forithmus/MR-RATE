import csv
import json
import math

import numpy as np
import pytest

from mrrate_report_training.clinical_metrics import (
    align_labels,
    compute_clinical_metrics,
    load_label_csv,
    write_metrics,
)


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_hand_computed_confusion():
    gt = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=float)
    pred = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)
    result = compute_clinical_metrics(gt, pred, ["a", "b"])
    block_a = result["per_pathology"]["a"]
    assert (block_a["tp"], block_a["fp"], block_a["tn"], block_a["fn"]) == (
        1,
        1,
        1,
        1,
    )
    assert block_a["sensitivity"] == pytest.approx(0.5)
    assert block_a["specificity"] == pytest.approx(0.5)
    assert block_a["f1"] == pytest.approx(0.5)
    assert block_a["accuracy"] == pytest.approx(0.5)
    block_b = result["per_pathology"]["b"]
    assert block_b["sensitivity"] == pytest.approx(1.0)
    assert block_b["specificity"] == pytest.approx(2.0 / 3.0)
    summary = result["summary"]
    assert summary["subset_accuracy"] == pytest.approx(0.5)
    # micro: tp=2 fp=2 tn=3 fn=1
    assert summary["micro"]["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert summary["micro"]["specificity"] == pytest.approx(3.0 / 5.0)
    assert summary["micro"]["accuracy"] == pytest.approx(5.0 / 8.0)


def test_perfect_prediction():
    gt = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    result = compute_clinical_metrics(gt, gt.copy(), ["a", "b", "c"])
    assert result["summary"]["subset_accuracy"] == 1.0
    assert result["summary"]["macro"]["f1"] == pytest.approx(1.0)
    assert result["summary"]["micro"]["specificity"] == pytest.approx(1.0)


def test_f1_zero_when_no_true_positives_but_defined():
    # tp=0, fp=1, fn=1: F1 is a well-defined 0, not NaN.
    gt = np.array([[1], [0]], dtype=float)
    pred = np.array([[0], [1]], dtype=float)
    result = compute_clinical_metrics(gt, pred, ["a"])
    assert result["per_pathology"]["a"]["f1"] == 0.0
    assert result["summary"]["macro"]["f1"] == 0.0


def test_weighted_skips_zero_support_classes():
    # Class b has no positives (weight 0, sensitivity NaN); the weighted
    # aggregate must still be defined from class a alone.
    gt = np.array([[1, 0], [1, 0]], dtype=float)
    pred = np.array([[1, 0], [0, 0]], dtype=float)
    result = compute_clinical_metrics(gt, pred, ["a", "b"])
    assert result["summary"]["weighted"]["sensitivity"] == pytest.approx(0.5)


def test_degenerate_class_is_nan_and_excluded_from_macro():
    gt = np.array([[1], [1]], dtype=float)  # no negatives: specificity undefined
    pred = np.array([[1], [0]], dtype=float)
    result = compute_clinical_metrics(gt, pred, ["a"])
    assert math.isnan(result["per_pathology"]["a"]["specificity"])
    assert result["summary"]["macro"]["specificity"] is None
    assert result["summary"]["macro"]["sensitivity"] == pytest.approx(0.5)


def test_schema_agnostic_87_pathologies():
    generator = np.random.default_rng(0)
    names = [f"pathology_{index:02d}" for index in range(87)]
    gt = generator.integers(0, 2, size=(20, 87)).astype(float)
    pred = generator.integers(0, 2, size=(20, 87)).astype(float)
    result = compute_clinical_metrics(gt, pred, names)
    assert len(result["per_pathology"]) == 87
    assert result["summary"]["pathologies"] == 87


def test_load_label_csv_rejects_bad_input(tmp_path):
    path = tmp_path / "labels.csv"
    write_csv(path, ["study_uid", "a"], [["s1", "2"]])
    with pytest.raises(ValueError, match="non-binary"):
        load_label_csv(path)
    write_csv(path, ["study_uid", "a"], [["s1", "1"], ["s1", "0"]])
    with pytest.raises(ValueError, match="duplicate"):
        load_label_csv(path)
    write_csv(path, ["other", "a"], [["s1", "1"]])
    with pytest.raises(ValueError, match="study_uid"):
        load_label_csv(path)


def test_align_labels_by_uid_and_column_name(tmp_path):
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    # Ground truth covers extra studies; prediction columns are reordered.
    write_csv(
        gt_path,
        ["study_uid", "a", "b"],
        [["s1", "1", "0"], ["s2", "0", "1"], ["s3", "1", "1"]],
    )
    write_csv(pred_path, ["study_uid", "b", "a"], [["s2", "1", "0"], ["s1", "0", "1"]])
    ids, names, gt, pred = align_labels(gt_path, pred_path)
    assert ids == ["s2", "s1"]
    assert names == ["a", "b"]
    assert gt.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert pred.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_align_labels_rejects_unknown_prediction(tmp_path):
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    write_csv(gt_path, ["study_uid", "a"], [["s1", "1"]])
    write_csv(pred_path, ["study_uid", "a"], [["sX", "1"]])
    with pytest.raises(ValueError, match="lack ground-truth"):
        align_labels(gt_path, pred_path)


def test_align_labels_rejects_schema_mismatch(tmp_path):
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    write_csv(gt_path, ["study_uid", "a"], [["s1", "1"]])
    write_csv(pred_path, ["study_uid", "b"], [["s1", "1"]])
    with pytest.raises(ValueError, match="schemas differ"):
        align_labels(gt_path, pred_path)


def test_write_metrics_serializes_nan_as_null(tmp_path):
    gt = np.array([[1], [1]], dtype=float)
    pred = np.array([[1], [0]], dtype=float)
    result = compute_clinical_metrics(gt, pred, ["a"])
    write_metrics(result, tmp_path)
    payload = json.loads((tmp_path / "clinical_metrics.json").read_text())
    assert payload["per_pathology"]["a"]["specificity"] is None
    rows = list(csv.DictReader((tmp_path / "per_pathology_metrics.csv").open()))
    assert rows[0]["pathology"] == "a"
    assert rows[0]["specificity"] == ""
