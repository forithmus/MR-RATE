import ast
import csv
import json

import pytest

from mrrate_report_training.clinical_metrics import load_label_csv
from mrrate_report_training.extract_labels import (
    _DEFAULT_CLASSIFIER_DIR,
    _DEFAULT_DIAGNOSES_JSON,
    _keyword_label,
    _resume_labels_dir,
    _stage_reports,
    extract_keyword_labels,
    extract_vllm_labels,
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


def test_load_neurovfm_diagnoses_format(tmp_path):
    diagnoses = tmp_path / "diagnoses.json"
    diagnoses.write_text(
        json.dumps(
            {
                "diagnoses": [
                    {"key": "acute_ischemic_stroke", "guidance": "x"},
                    {
                        "key": "brain_abscess",
                        "guidance": "y",
                        "synonyms": ["cerebral abscess"],
                    },
                ]
            }
        )
    )
    schema = load_pathology_schema(diagnoses)
    assert list(schema) == ["acute_ischemic_stroke", "brain_abscess"]
    assert schema["acute_ischemic_stroke"] == [
        "acute_ischemic_stroke",
        "acute ischemic stroke",
    ]
    assert schema["brain_abscess"][-1] == "cerebral abscess"


def test_bundled_extractor_is_current_neurovfm74_pipeline():
    schema = load_pathology_schema(_DEFAULT_DIAGNOSES_JSON)
    assert len(schema) == 74
    assert list(schema)[:3] == [
        "subdural_hematoma",
        "epidural_hematoma",
        "brain_contusion",
    ]
    assert list(schema)[-1] == "spine_syrinx"
    assert (_DEFAULT_CLASSIFIER_DIR / "extract_neurovfm_dx_gemma.py").is_file()
    assert (_DEFAULT_CLASSIFIER_DIR / "merge_labels.py").is_file()


def test_bundled_prompt_uses_neurovfm_classification_rules():
    source = (_DEFAULT_CLASSIFIER_DIR / "extract_neurovfm_dx_gemma.py").read_text()
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "NEUROVFM_PROMPT"
            for target in node.targets
        )
    )
    prompt = ast.literal_eval(assignment.value)
    assert "ONLY base the classification on the FINDINGS and IMPRESSION" in prompt
    assert "mentioned as a possible diagnosis" in prompt
    assert '"rationale"' in prompt and '"label"' in prompt
    assert "STEP1_PROMPT" not in source


def test_vllm_backend_rejects_legacy_pathology_schema(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"pathologies": {"Gliosis": {}}}))
    with pytest.raises(ValueError, match="NeuroVFM"):
        extract_vllm_labels(
            [{"study_uid": "s1", "findings_pred": "Gliosis."}],
            load_pathology_schema(legacy),
            diagnoses_json=legacy,
            work_dir=tmp_path / "work",
            classifier_dir=_DEFAULT_CLASSIFIER_DIR,
            text_column="findings_pred",
            model_name=None,
            batch_size=1,
            seed=42,
        )


def test_vllm_wrapper_invokes_neurovfm_extractor_and_merges(tmp_path, monkeypatch):
    diagnoses = tmp_path / "diagnoses.json"
    diagnoses.write_text(
        json.dumps(
            {
                "diagnoses": [
                    {"key": "acute_ischemic_stroke", "guidance": ""},
                    {"key": "brain_abscess", "guidance": ""},
                ]
            }
        )
    )
    classifier_dir = tmp_path / "classifier"
    classifier_dir.mkdir()
    (classifier_dir / "extract_neurovfm_dx_gemma.py").write_text("")
    (classifier_dir / "merge_labels.py").write_text("")
    calls = []

    def fake_run(command, *, check, env):
        assert check is True
        calls.append(command)
        if command[1].endswith("merge_labels.py"):
            output = command[command.index("--output") + 1]
            with open(output, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["study_uid", "acute_ischemic_stroke", "brain_abscess"]
                )
                writer.writerow(["s1", 1, 0])

    monkeypatch.setattr("mrrate_report_training.extract_labels.subprocess.run", fake_run)
    rows = [
        {"study_uid": "s1", "findings_pred": "Acute infarct."},
        {"study_uid": "s2", "findings_pred": ""},
    ]
    labeled = extract_vllm_labels(
        rows,
        load_pathology_schema(diagnoses),
        diagnoses_json=diagnoses,
        work_dir=tmp_path / "work",
        classifier_dir=classifier_dir,
        text_column="findings_pred",
        model_name="google/gemma-4-31B-it",
        batch_size=2,
        seed=42,
    )
    assert calls[0][1].endswith("extract_neurovfm_dx_gemma.py")
    assert "--diagnoses_json" in calls[0]
    assert "--pathologies_json" not in calls[0]
    assert labeled == [
        {
            "study_uid": "s1",
            "labels": {"acute_ischemic_stroke": 1, "brain_abscess": 0},
        },
        {
            "study_uid": "s2",
            "labels": {"acute_ischemic_stroke": 0, "brain_abscess": 0},
        },
    ]


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


def test_resume_directory_refuses_changed_report_text(tmp_path):
    rows = [{"study_uid": "s1", "findings_pred": "Original report."}]
    labels_dir = _resume_labels_dir(
        rows,
        work_dir=tmp_path,
        text_column="findings_pred",
        diagnoses_json=_DEFAULT_DIAGNOSES_JSON,
        model_name=None,
        seed=42,
    )
    (labels_dir / "labels_rank_0.json").write_text("{}")
    assert (
        _resume_labels_dir(
            rows,
            work_dir=tmp_path,
            text_column="findings_pred",
            diagnoses_json=_DEFAULT_DIAGNOSES_JSON,
            model_name=None,
            seed=42,
        )
        == labels_dir
    )
    changed = [{"study_uid": "s1", "findings_pred": "Changed report."}]
    with pytest.raises(ValueError, match="different reports"):
        _resume_labels_dir(
            changed,
            work_dir=tmp_path,
            text_column="findings_pred",
            diagnoses_json=_DEFAULT_DIAGNOSES_JSON,
            model_name=None,
            seed=42,
        )
