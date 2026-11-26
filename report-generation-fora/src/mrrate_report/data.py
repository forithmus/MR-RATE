"""Studies, report targets, labels and the threaded loader of the MR writer.

A case is one study: ``{"study_uid", "image_paths", "split"}`` (discovered in the MR-RATE atlas tree). The target is
the ``findings`` field of the study's report (the full ``report`` text when the report has no findings field).
"""
from __future__ import annotations

import csv
import glob
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

csv.field_size_limit(10 ** 9)
EMPTY_FINDINGS = "No findings were reported."


def read_cases(data_folder: str, splits_csv: str, splits, space: str) -> list[dict]:
    from mr_dino.data import discover_raw_aligned
    with open(splits_csv, newline="") as handle:
        split_of = {r["study_uid"]: r["split"] for r in csv.DictReader(handle) if r["split"] in set(splits)}
    cases = discover_raw_aligned(data_folder, set(split_of), space)
    return sorted(({**c, "split": split_of[c["study_uid"]]} for c in cases), key=lambda c: c["study_uid"])


def read_reports(pattern: str) -> dict[str, str]:
    """study_uid -> findings text from the report CSV files (glob)."""
    out = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no report CSV matches {pattern}")
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                text = (r.get("findings") or "").strip() or (r.get("report") or "").strip()
                out[str(r["study_uid"])] = text or EMPTY_FINDINGS
    return out


def read_positives(labels_csv: str, classes: list[str]) -> dict[str, np.ndarray]:
    """study_uid -> indices of the diagnoses labelled 1."""
    out = {}
    with open(labels_csv, newline="") as handle:
        for r in csv.DictReader(handle):
            out[str(r["study_uid"])] = np.array([j for j, c in enumerate(classes) if r.get(c, "").strip() in ("1", "1.0")],
                                                np.int64)
    return out


class DenseLogits:
    """Precomputed study logits (``calibrate --dense``), checked against the calibration."""

    def __init__(self, path: str, labels: list[str], calibration: dict):
        from pathlib import Path
        if Path(str(calibration.get("source_predictions", ""))).resolve() != Path(path).resolve():
            raise ValueError("dense logits are not the source predictions of the calibration")
        z = np.load(path, allow_pickle=False)
        if [str(c) for c in z["label"].tolist()] != list(labels):
            raise ValueError("dense logits vocabulary differs from the calibration labels")
        self.logits = np.asarray(z["logit"], np.float32)
        self.row = {str(u): k for k, u in enumerate(z["report_uid"].tolist())}

    def __contains__(self, uid) -> bool:
        return str(uid) in self.row

    def get(self, uid: str, device) -> torch.Tensor:
        return torch.tensor(self.logits[self.row[str(uid)]], device=device).unsqueeze(0)


class Prefetcher:
    """Thread prefetch of (case, volumes, findings) in this rank's order. An unreadable study is replaced by the next
    case of the list (logged), so every rank still runs one forward per step."""

    def __init__(self, stream, cases: list[dict], order, reports: dict, depth: int = 3, workers: int = 3):
        self.stream, self.cases, self.order, self.reports = stream, cases, list(order), reports
        self.pool = ThreadPoolExecutor(workers)
        self.depth, self.pending, self.next_submit = depth, {}, 0
        for _ in range(depth):
            self._submit()

    def _load(self, index):
        for attempt in range(8):
            case = self.cases[(int(index) + attempt) % len(self.cases)]
            try:
                return case, self.stream.prepare(case), self.reports.get(case["study_uid"], EMPTY_FINDINGS)
            except Exception as error:
                print(json.dumps(dict(event="case_skip", case=case["study_uid"], error=repr(error)[:200])), flush=True)
        raise RuntimeError(f"8 consecutive unreadable studies from item {index}")

    def _submit(self):
        if self.next_submit < len(self.order):
            k = self.next_submit
            self.pending[k] = self.pool.submit(self._load, self.order[k])
            self.next_submit += 1

    def get(self, k):
        future = self.pending.pop(k)
        self._submit()
        try:
            return future.result(timeout=900)
        except Exception as error:
            print(json.dumps(dict(event="loader_retry", item=int(self.order[k]), error=repr(error)[:200])), flush=True)
            return self._load(self.order[k])

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
