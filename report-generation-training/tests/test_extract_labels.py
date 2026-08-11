import csv
import json

import pytest

from mrrate_report_training.clinical_metrics import load_label_csv
from mrrate_report_training.extract_labels import (
    _keyword_label,
    _stage_reports,
    extract_keyword_labels,
    load_pathology_schema,
    write_labels_csv,
)


def test_keyword_label_matches_and_negates():
    phrases = ["cerebral infarction", "infarct"]
    assert _keyword_label("A chronic infarct is present.", phrases) == 1
    assert _keyword_label("There is no infarct.", phrases) == 0
    assert _keyword_label("Findings without infarct or edema.", phrases) == 0
    assert _keyword_label("Cerebral infarction is seen.", phrases) == 1
    assert _keyword_label("The study is unremarkable.", phrases) == 0
    # Negation window is bounded: distant negation does not suppress.
    assert (
        _keyword_label(
            "no evidence of hemorrhage but there is an acute infarct",
            phrases,
        )
        == 1
    )
    # Negation never leaks across a sentence boundary.
    assert _keyword_label("There is no edema. There is infarct.", phrases) == 1
    assert _keyword_label("There is no edema.\nInfarct is seen.", phrases) == 1


def test_load_pathology_schema_formats(tmp_path):
    structured = tmp_path / "structured.json"
    structured.write_text(
        json.dumps(
            {
                "pathologies": {
                    "Cerebral infarction": {
                        "positive": "There is infarct",
                        "negative": "There is no infarct",
                        "synonyms": ["infarct"],
                    },
                    "Gliosis": {"positive": "x", "negative": "y"},
                }
            }
        )
    )
    schema = load_pathology_schema(structured)
    assert schema["Cerebral infarction"] == ["Cerebral infarction", "infarct"]
    assert schema["Gliosis"] == ["Gliosis"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(["A", "B"]))
    assert list(load_pathology_schema(legacy)) == ["A", "B"]
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"pathologies": {}}))
    with pytest.raises(ValueError):
        load_pathology_schema(empty)


def test_extract_and_write_round_trip(tmp_path):
    schema = {
        "Cerebral infarction": ["Cerebral infarction", "infarct"],
        "Cerebral hemorrhage": ["Cerebral hemorrhage", "hemorrhage"],
    }
    rows = [
        {"study_uid": "s2", "findings_pred": "There is an acute infarct."},
        {"study_uid": "s1", "findings_pred": "No hemorrhage. No infarct."},
    ]
    labeled = extract_keyword_labels(rows, schema, text_column="findings_pred")
    output = tmp_path / "labels.csv"
    write_labels_csv(labeled, list(schema), output)
    ids, names, matrix = load_label_csv(output)
    assert ids == ["s1", "s2"]  # sorted on write
    assert names == list(schema)
    assert matrix.tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_stage_reports_separates_empty_findings(tmp_path):
    rows = [
        {"study_uid": "s1", "findings_pred": "An infarct."},
        {"study_uid": "s2", "findings_pred": "   "},
    ]
    reports_dir, empty = _stage_reports(rows, tmp_path, "findings_pred")
    assert empty == ["s2"]
    staged = list(csv.DictReader((reports_dir / "batch00_reports.csv").open()))
    assert [row["study_uid"] for row in staged] == ["s1"]
    assert staged[0]["findings"] == "An infarct."
