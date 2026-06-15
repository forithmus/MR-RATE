"""Build the neuroradiologist's merged-group labels in a SINGLE combined CSV,
recomputing the Opus(Claude) AND GPT(OpenAI) agreement from the RAW prediction
JSONs over ALL 37 pathologies (so the 5 pathologies dropped in
build_agreement_splits.py are recovered, not ignored).

Two grouping schemes (from Bene, neuroradiology):
  1) Pathophysiologie  -> 8 groups, columns prefixed "PP_"
  2) Bildphaenotyp     -> 6 groups, columns prefixed "BP_"

A study is positive for a group iff it is positive for ANY member pathology
(logical OR). Per-pathology agreement uses the same strict AND rule as
build_agreement_splits.py:
  both 1 -> 1 | any 0 -> 0 | one side missing -> present side | both missing -> 0(blank)

The two pathologies ungrouped in BOTH schemes (Empty sella syndrome,
Hyperostosis of skull) are intentionally excluded.

Inputs (same sources as build_agreement_splits.py, but NOTHING is dropped):
  eval_set_predictions_5k.json                 -> claude_labels + gpt_labels
  ../../../remaining_eval/eval_set_predictions_chunk_0{0..4}.json
                                               -> nvidia_opus47_labels + nvidia_gpt55_labels
  splits_hf.csv                                -> study_uid -> split

Outputs (under splits_merged/):
  mrrate_merged_labels.csv  -- study_uid + 14 binary group cols
  pathologies.json          -- same 14 names, ready for --pathologies_file
  group_definitions.json    -- group -> members (all present now)
  splits.csv                -- batch_id,patient_uid,study_uid,split (kept UIDs only)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
PRED_5K = HERE / "eval_set_predictions_5k.json"
CHUNK_DIR = Path("/hnvme/workspace/b180dc51-sezgin/MR-RATE/remaining_eval")
# 92k chunks augmented with a 3rd model (Nemotron) for the "majority" mode.
CHUNK_DIR_NEM = Path("/hnvme/workspace/b180dc51-sezgin/remaining_eval_with_nemotron-2")
SPLITS_CSV = HERE / "splits_hf.csv"
# 32-column agreement labels (5 pathologies dropped) for the legacy "csv32" mode.
CSV32_LABELS = HERE / "splits_agreement" / "mrrate_labels.csv"
CSV32_SPLITS = HERE / "splits_agreement" / "splits.csv"

PP_GROUPS = [
    ("PP_Cerebrovascular", ["Cerebral infarction", "Cerebral hemorrhage", "Lacunar infarct",
        "Silent micro-hemorrhage of brain", "Cavernous hemangioma",
        "Subdural intracranial hemorrhage", "Intracranial aneurysm", "Watershed infarct"]),
    ("PP_Neoplastic", ["Metastatic malignant neoplasm to brain", "Intracranial meningioma",
        "Glioma", "Pituitary adenoma", "Schwannoma"]),
    ("PP_Neurodegenerative", ["Cerebral atrophy", "Ventriculomegaly", "Cerebellar degeneration"]),
    ("PP_Spinal", ["Herniation of nucleus pulposus", "Spinal cord compression",
        "Foraminal Spinal Stenosis", "Spinal stenosis", "Hemangioma of vertebral column"]),
    ("PP_Cystic_developmental", ["Arachnoid cyst", "Cyst of pineal gland",
        "Structure of cave of septum pellucidum", "Mega cisterna magna", "Chiari malformation",
        "Rathke's pouch cyst", "Choroid plexus cyst", "Lipoma of brain"]),
    ("PP_Infectious", ["Mastoiditis", "Chronic mastoiditis"]),
    ("PP_Inflammatory", ["Demyelinating disease of central nervous system"]),
    ("PP_Unspecific_bucket", ["Gliosis", "Cerebral edema", "Encephalomalacia"]),
]

BP_GROUPS = [
    ("BP_Atrophies", ["Cerebral atrophy", "Ventriculomegaly", "Cerebellar degeneration"]),
    ("BP_Contrast_enhancing_intracranial", ["Metastatic malignant neoplasm to brain",
        "Intracranial meningioma", "Glioma", "Pituitary adenoma", "Schwannoma"]),
    ("BP_Infectious_lesions", ["Mastoiditis", "Chronic mastoiditis"]),
    ("BP_Edematous_lesions", ["Cerebral infarction", "Lacunar infarct", "Watershed infarct",
        "Demyelinating disease of central nervous system"]),
    ("BP_Hemorrhagic_lesions", ["Cerebral hemorrhage", "Silent micro-hemorrhage of brain",
        "Cavernous hemangioma", "Encephalomalacia"]),
    ("BP_Cystic_lesions", ["Arachnoid cyst", "Cyst of pineal gland",
        "Structure of cave of septum pellucidum", "Mega cisterna magna",
        "Rathke's pouch cyst", "Choroid plexus cyst"]),
]

ALL_GROUPS = PP_GROUPS + BP_GROUPS


def agreement(a: dict | None, b: dict | None, p: str) -> int:
    """Strict AND per pathology. Returns 1 (agree positive) else 0.

    both1 -> 1 | any0 -> 0 | one missing -> present side | both missing -> 0.
    """
    a = a or {}
    b = b or {}
    x, y = a.get(p), b.get(p)
    if x is None and y is None:
        return 0
    if x is None:
        return 1 if y == 1 else 0
    if y is None:
        return 1 if x == 1 else 0
    return 1 if (x == 1 and y == 1) else 0


def load_split_map(path: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            m[r["study_uid"]] = {
                "split": r["split"],
                "batch_id": r.get("batch_id", ""),
                "patient_uid": r.get("patient_uid", ""),
            }
    return m


def write_outputs(out_dir: Path, rows: list[tuple[str, list[int]]], split_map: dict) -> None:
    """rows = [(study_uid, [group values...])]; writes the 4 artifacts."""
    out_dir.mkdir(exist_ok=True)
    out_cols = [name for name, _ in ALL_GROUPS]
    counts = {c: 0 for c in out_cols}
    kept_uids: set[str] = set()
    n_split = {"train": 0, "val": 0, "test": 0}

    out_path = out_dir / "mrrate_merged_labels.csv"
    per_split: dict[str, list[list]] = {"train": [], "val": [], "test": []}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_uid"] + out_cols)
        for uid, vals in rows:
            w.writerow([uid] + vals)
            kept_uids.add(uid)
            split = split_map[uid]["split"]
            n_split[split] += 1
            per_split[split].append([uid] + vals)
            for name, v in zip(out_cols, vals):
                counts[name] += v

    # Per-split CSVs (train.csv / val.csv / test.csv), same columns as the
    # combined labels file.
    for split, srows in per_split.items():
        fname = "val.csv" if split == "val" else f"{split}.csv"
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["study_uid"] + out_cols)
            w.writerows(srows)

    paths_json = {}
    for name, _ in ALL_GROUPS:
        label = name.replace("PP_", "").replace("BP_", "").replace("_", " ")
        paths_json[name] = {"positive": f"There is {label}", "negative": f"There is no {label}"}
    (out_dir / "pathologies.json").write_text(json.dumps({"pathologies": paths_json}, indent=2))
    (out_dir / "group_definitions.json").write_text(
        json.dumps({name: ms for name, ms in ALL_GROUPS}, indent=2)
    )
    with open(out_dir / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_id", "patient_uid", "study_uid", "split"])
        for uid, meta in split_map.items():
            if uid in kept_uids:
                w.writerow([meta["batch_id"], meta["patient_uid"], uid, meta["split"]])

    n = len(kept_uids)
    print(f"\nWrote {out_path}  ({n:,} studies, {len(out_cols)} label columns)")
    print(f"  kept per split: {n_split}")
    print(f"{'column':40s} {'pos':>7s} {'%':>6s}")
    for name, _ in ALL_GROUPS:
        print(f"{name:40s} {counts[name]:7d} {100*counts[name]/n:5.1f}")


def build_from_csv32(out_dir: Path) -> None:
    """Legacy mode: OR-collapse the existing 32-column agreement labels.
    The 5 dropped pathologies are absent and contribute nothing."""
    split_map = load_split_map(CSV32_SPLITS)
    with open(CSV32_LABELS) as f:
        reader = csv.DictReader(f)
        present = set(reader.fieldnames) - {"study_uid"}
        rows = []
        for row in reader:
            uid = row["study_uid"]
            if uid not in split_map:
                continue
            vals = []
            for _, ms in ALL_GROUPS:
                used = [m for m in ms if m in present]
                vals.append(1 if any(float(row[m]) > 0 for m in used) else 0)
            rows.append((uid, vals))
    write_outputs(out_dir, rows, split_map)


def build_from_raw(out_dir: Path) -> None:
    split_map = load_split_map(SPLITS_CSV)

    pred_5k = json.loads(PRED_5K.read_text())
    paths_all = pred_5k["pathologies"]  # 37, in canonical order

    # Validate every group member is a real pathology.
    members = {m for _, ms in ALL_GROUPS for m in ms}
    unknown = members - set(paths_all)
    if unknown:
        raise RuntimeError(f"Unknown group members: {sorted(unknown)}")

    # study_uid -> (claude/opus labels, gpt labels). 5k wins over chunks.
    pred: dict[str, tuple[dict, dict]] = {}
    for it in pred_5k["items"]:
        pred[it["study_uid"]] = (it.get("claude_labels") or {}, it.get("gpt_labels") or {})
    print(f"loaded 5k preds:    {len(pred):,}")

    for i in range(5):
        chunk = json.loads((CHUNK_DIR / f"eval_set_predictions_chunk_0{i}.json").read_text())
        assert chunk["pathologies"] == paths_all, f"pathology mismatch in chunk {i}"
        for it in chunk["items"]:
            uid = it["study_uid"]
            if uid in pred:
                continue
            pred[uid] = (it.get("nvidia_opus47_labels") or {}, it.get("nvidia_gpt55_labels") or {})
        print(f"loaded chunk_0{i}: cumulative {len(pred):,}")

    rows = []
    no_pred = {"train": 0, "val": 0, "test": 0}
    for uid, meta in split_map.items():
        if uid not in pred:
            no_pred[meta["split"]] += 1
            continue
        a, b = pred[uid]
        agree = {p: agreement(a, b, p) for p in members}
        vals = [1 if any(agree[m] for m in ms) else 0 for _, ms in ALL_GROUPS]
        rows.append((uid, vals))
    print(f"no-pred skipped: {no_pred}")
    write_outputs(out_dir, rows, split_map)


def majority(votes: list, p: str) -> int:
    """Strict majority of PRESENT votes for pathology p.
    pos*2 > present -> 1. 3 present -> need 2; 2 present -> need both; 1 -> need it."""
    present = [d.get(p) for d in votes if (d or {}).get(p) is not None]
    if not present:
        return 0
    pos = sum(1 for v in present if v == 1)
    return 1 if pos * 2 > len(present) else 0


def build_majority(out_dir: Path) -> None:
    """3-way majority of Claude(Opus) + GPT + Nemotron, then OR into groups.
    5k uses claude/gpt/nemotron; 92k uses the nemotron-augmented chunks."""
    split_map = load_split_map(SPLITS_CSV)
    pred_5k = json.loads(PRED_5K.read_text())
    paths_all = pred_5k["pathologies"]
    members = {m for _, ms in ALL_GROUPS for m in ms}

    # study_uid -> (claude/opus, gpt, nemotron). 5k wins over chunks.
    pred: dict[str, tuple[dict, dict, dict]] = {}
    n_models = {0: 0, 1: 0, 2: 0, 3: 0}
    for it in pred_5k["items"]:
        pred[it["study_uid"]] = (
            it.get("claude_labels") or {},
            it.get("gpt_labels") or {},
            it.get("nemotron_labels") or {},
        )
    print(f"loaded 5k preds:    {len(pred):,}")

    for i in range(5):
        chunk = json.loads((CHUNK_DIR_NEM / f"eval_set_predictions_chunk_0{i}.json").read_text())
        assert chunk["pathologies"] == paths_all, f"pathology mismatch in chunk {i}"
        for it in chunk["items"]:
            uid = it["study_uid"]
            if uid in pred:
                continue
            pred[uid] = (
                it.get("nvidia_opus47_labels") or {},
                it.get("nvidia_gpt55_labels") or {},
                it.get("nvidia_nemotron3_super_v3_labels") or {},
            )
        print(f"loaded chunk_0{i}: cumulative {len(pred):,}")

    rows = []
    no_pred = {"train": 0, "val": 0, "test": 0}
    for uid, meta in split_map.items():
        if uid not in pred:
            no_pred[meta["split"]] += 1
            continue
        votes = pred[uid]
        n_models[sum(1 for d in votes if d)] += 1
        maj = {p: majority(votes, p) for p in members}
        vals = [1 if any(maj[m] for m in ms) else 0 for _, ms in ALL_GROUPS]
        rows.append((uid, vals))
    print(f"no-pred skipped: {no_pred}")
    print(f"models available per study: {n_models}  (3 = full 3-way majority)")
    write_outputs(out_dir, rows, split_map)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["raw", "csv32", "majority"], default="raw",
                    help="raw = Claude AND GPT agreement over all 37 pathologies (default); "
                         "csv32 = collapse the existing 32-col labels (5 paths stay dropped); "
                         "majority = 3-way majority of Claude + GPT + Nemotron.")
    ap.add_argument("--out-dir", default=None, help="output folder (default per source).")
    a = ap.parse_args()
    defaults = {"raw": "splits_merged", "csv32": "splits_merged_32col",
                "majority": "splits_merged_majority"}
    out = Path(a.out_dir) if a.out_dir else HERE / defaults[a.source]
    {"raw": build_from_raw, "csv32": build_from_csv32, "majority": build_majority}[a.source](out)
