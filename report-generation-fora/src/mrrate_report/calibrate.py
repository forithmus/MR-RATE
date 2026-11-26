"""Plan calibration of the MR writer from TRAIN predictions only (the FORA plan rules).

* Platt per diagnosis, p = sigmoid(a * logit + b), ridge prior toward (1, 0) (l2 = 1.0), fitted on the train study
  logits against the train labels; unknown labels (empty or not 0/1) are left out per diagnosis.
* possible tier = per-diagnosis Youden cutoff on the train raw probabilities;
* confident tier = calibrated p >= 0.5, never below the Youden cutoff;
* base rate in the plan = train positives / train labelled studies.
Validation and test studies are never used. Outputs the calibration JSON read by ``plan.MRPlanBuilder`` and the dense
logits (every split) read by the trainers.

python -m mrrate_report.calibrate --pred PRED_DIR --labels-csv L --out calibration.json --dense dense_logits.npz
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from .schema import label_order_sha256, readable


def load_predictions(pred_dir: str):
    uids, splits, logits = [], [], []
    for f in sorted(glob.glob(f"{pred_dir}/shard_r*.npz")):
        d = np.load(f)
        uids += list(map(str, d["uids"]))
        splits += list(map(str, d["splits"]))
        logits.append(d["logits"])
    if not uids:
        raise FileNotFoundError(f"no shards in {pred_dir}")
    if len(set(uids)) != len(uids):
        raise ValueError("a study appears in two prediction shards")
    info = json.loads(Path(pred_dir, "classes.json").read_text())
    return np.array(uids), np.array(splits), np.concatenate(logits).astype(np.float64), info


def read_labels(path: str, classes: list[str], uids) -> np.ndarray:
    """[n, C] with 1 / 0 and NaN for unknown."""
    rows = {}
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            rows[r["study_uid"]] = r
    Y = np.full((len(uids), len(classes)), np.nan)
    for i, u in enumerate(uids):
        r = rows.get(u)
        if r is None:
            continue
        for j, c in enumerate(classes):
            try:
                v = float(r.get(c, ""))
            except ValueError:
                continue
            if v in (0.0, 1.0):
                Y[i, j] = v
    return Y


def fit_platt(x, y, mask, l2: float = 1.0, iterations: int = 50):
    """Per-class Platt fit, Newton with backtracking, ridge toward (1, 0); returns (slope, intercept)."""
    x = x.astype(np.float64)
    y = np.nan_to_num(y).astype(np.float64)
    w = mask.astype(np.float64)
    C = x.shape[1]
    a, b = np.ones(C), np.zeros(C)

    def objective(a_, b_):
        z = x * a_ + b_
        return ((np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))) * w).sum(0) + 0.5 * l2 * ((a_ - 1) ** 2 + b_ ** 2)

    current = objective(a, b)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x * a + b, -40, 40)))
        r, s = (p - y) * w, p * (1 - p) * w
        ga, gb = (r * x).sum(0) + l2 * (a - 1), r.sum(0) + l2 * b
        haa, hab, hbb = (s * x * x).sum(0) + l2, (s * x).sum(0), s.sum(0) + l2
        det = haa * hbb - hab * hab
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step, accepted = np.ones(C), np.zeros(C, bool)
        for _h in range(30):
            trial = objective(a - step * da, b - step * db)
            better = (trial <= current) & ~accepted
            a, b = np.where(better, a - step * da, a), np.where(better, b - step * db, b)
            current, accepted = np.where(better, trial, current), accepted | better
            if accepted.all():
                break
            step = np.where(accepted, step, step * 0.5)
        if max(np.abs(step * da).max(), np.abs(step * db).max()) < 1e-7:
            break
    return np.maximum(a, 0.05), b


def youden(s, y) -> float:
    """Youden cutoff (positive if score >= cut) at distinct-score boundaries."""
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]
    j = tp / y.sum() - fp / (len(y) - y.sum())
    j[~last] = -np.inf
    return float(s[int(np.argmax(j))])


def tiers(slope, intercept, youden_raw):
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))  # noqa: E731
    raw_p50 = 1 / (1 + np.exp(intercept / np.maximum(slope, 1e-6)))
    raw_plan = np.maximum(raw_p50, youden_raw)
    raw_possible = np.minimum(raw_plan, youden_raw)
    plan = 1 / (1 + np.exp(-(slope * logit(raw_plan) + intercept)))
    possible = np.minimum(1 / (1 + np.exp(-(slope * logit(raw_possible) + intercept))), plan)
    return raw_plan, raw_possible, plan, possible


def build(uids, splits, X, Y, classes, l2=1.0):
    train = splits == "train"
    known = np.isfinite(Y) & train[:, None]
    slope, intercept = fit_platt(X, Y, known, l2)
    raw = 1 / (1 + np.exp(-X))
    cut = np.full(len(classes), 0.5)
    positives = np.zeros(len(classes), int)
    labelled = np.zeros(len(classes), int)
    for j in range(len(classes)):
        m = known[:, j]
        y = Y[m, j]
        positives[j], labelled[j] = int(y.sum()), int(m.sum())
        if 0 < y.sum() < len(y):
            cut[j] = youden(raw[m, j], y)
    raw_plan, raw_possible, plan, possible = tiers(slope, intercept, cut)
    never = positives == 0          # no train positive: never planned
    plan[never], possible[never] = 2.0, 2.0
    return dict(slope=slope, intercept=intercept, raw_plan=raw_plan, raw_possible=raw_possible, plan=plan,
                possible=possible, positives=positives, labelled=labelled, n_train=int(train.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="prediction directory of mrrate_report.predict")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--platt-l2", type=float, default=1.0)
    a = ap.parse_args(argv)
    uids, splits, X, info = load_predictions(a.pred)
    classes = list(info["classes"])
    Y = read_labels(a.labels_csv, classes, uids)
    c = build(uids, splits, X, Y, classes, a.platt_l2)
    dense = Path(a.dense).resolve()
    dense.parent.mkdir(parents=True, exist_ok=True)
    tmp = dense.with_name(dense.name + ".tmp.npz")
    np.savez(tmp, report_uid=uids, logit=X.astype(np.float32), label=np.array(classes), split=splits,
             class_index=np.arange(len(classes)))
    tmp.replace(dense)
    sel = splits == "val"
    raw_val = 1 / (1 + np.exp(-X[sel]))
    calibration = {
        "format": "mrrate_frozen_mil_train_platt", "format_version": 1, "mil_checkpoint": info.get("mil_checkpoint"),
        "source_predictions": str(dense), "source_split": "train", "n_calibration_studies": c["n_train"],
        "platt": {"l2_toward_identity": a.platt_l2, "min_slope": 0.05},
        "threshold_policy": {"plan": "calibrated probability (train Platt) >= 0.5, never below the train Youden cutoff",
                             "possible": "train Youden cutoff, never above the plan threshold"},
        "labels": classes, "label_order_sha256": label_order_sha256(classes),
        "canonical_plan_labels": [readable(k) for k in classes],
        "class_val_positive": c["positives"].tolist(), "class_labelled": c["labelled"].tolist(),
        "class_slope": c["slope"].tolist(), "class_intercept": c["intercept"].tolist(),
        "class_raw_plan_threshold": c["raw_plan"].tolist(), "class_raw_possible_threshold": c["raw_possible"].tolist(),
        "class_plan_threshold": c["plan"].tolist(), "class_possible_threshold": c["possible"].tolist(),
        "summary": {"val_confident_per_study": float((raw_val >= c["raw_plan"]).sum(1).mean()) if sel.any() else None,
                    "val_possible_per_study": float(((raw_val >= c["raw_possible"]) & (raw_val < c["raw_plan"])).sum(1).mean())
                    if sel.any() else None},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(calibration, indent=1))
    tmp.replace(out)
    print(json.dumps({"calibration": str(out), "dense": str(dense), **calibration["summary"]}), flush=True)


if __name__ == "__main__":
    main()
