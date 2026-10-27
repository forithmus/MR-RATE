"""Load the final ordered 74-diagnosis MRI label schema."""
import json
from pathlib import Path
from .probe import read_labels

DEFAULT_SCHEMA = (Path(__file__).resolve().parents[2] /
    "data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/"
    "07_mri_diagnosis_extraction/data/mri_diagnoses_74.json")


def read_final_labels(labels_csv, schema_path=DEFAULT_SCHEMA):
    schema=json.loads(Path(schema_path).read_text())
    names=[item["key"] for item in schema["diagnoses"]]
    if len(names)!=74 or len(set(names))!=74:
        raise ValueError("Production MRI MIL requires 74 unique diagnosis keys")
    columns,labels=read_labels(Path(labels_csv))
    missing=set(names)-set(columns);extra=set(columns)-set(names)
    if missing or extra or len(columns)!=74:
        raise ValueError(f"Final 74-class label mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    order=[columns.index(name) for name in names]
    return names,{uid:values[order] for uid,values in labels.items()}
