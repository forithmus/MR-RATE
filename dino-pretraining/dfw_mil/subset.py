"""Nested patient-level subsets of the training split for the data-size ablation.

Training patients are ordered once by a seeded hash of their id. A fraction ``f`` keeps the
first ``ceil(f * n_patients)`` patients of that order, so for one seed every smaller subset is
contained in every larger one, and all studies of a patient are kept or dropped together.
Validation and test are never subsetted.
"""
from __future__ import annotations

import hashlib
import math


def patient_order(patients, seed: int) -> list[str]:
    key = lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()
    return sorted(set(patients), key=lambda p: (key(p), p))


def train_patient_subset(train_patients, fraction: float, seed: int) -> set[str]:
    if not 0 < fraction <= 1:
        raise ValueError(f"--train-fraction must be in (0, 1], got {fraction}")
    order = patient_order(train_patients, seed)
    return set(order[:math.ceil(fraction * len(order))])


def subset_train_ids(train_ids, patient_of: dict, fraction: float, seed: int) -> list[str]:
    """``train_ids`` (any order) restricted to the patients of the nested subset."""
    if fraction >= 1:
        return list(train_ids)
    train_patients = {patient_of[u] for u in train_ids}
    keep = train_patient_subset(train_patients, fraction, seed)
    return [u for u in train_ids if patient_of[u] in keep]
