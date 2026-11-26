#!/usr/bin/env python3
"""Synthetic DPO pairs for the MR writer from TRAIN reports (the FORA X-ray pair rules on brain MRI).

chosen   = the study's findings text (the SFT target); its sha256 is stored per pair and re-checked by the trainer.
rejected = an edit of that text (relative to the written report, not to image truth), using the sentence tags of
           ``tag_sentences`` (abnormal sentence = one that states a vocabulary diagnosis as present or possible):
  omission           one abnormal sentence deleted
  multi_omission     2-4 abnormal sentences deleted (reports with at least 3)
  insertion_plan_fp  a planned diagnosis of the study (calibrated p >= its possible threshold, i.e. shown to the writer)
                     that the report neither asserts nor hedges and whose label is not positive is asserted by a donor
                     sentence of another TRAIN patient that asserts only that diagnosis; inserted after the last
                     abnormal sentence (at the start when there is none). At most 3 % of the studies per diagnosis.
  insertion_random   the same insertion for a random diagnosis that is not positive (square-root prevalence weights)
  normalised         every abnormal sentence removed and one frequent normal sentence of the corpus put in place of
                     the first
  repetition         a sentence repeated, or the last one or two sentences looped two or three times
Incidental and age-related diagnoses (``NO_INSERT``) are never inserted.
Output: one sqlite (tables metadata, pairs), read by ``train_dpo``.

python -m mrrate_report.dpo_pairs --tags TAG_DIR --calibration C --splits-csv S --labels-csv L --out pairs.sqlite
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .tag_sentences import sha256_text

SCHEMA = "mrrate_report_preference_index_v1"
TYPES = ("omission", "multi_omission", "insertion_plan_fp", "insertion_random", "normalised", "repetition")
NO_INSERT = {"cavum_septum_pellucidum", "mega_cisterna_magna", "pineal_cyst", "spinal_degenerative_changes",
             "cerebral_atrophy", "small_vessel_ischemic_disease"}
INSERT_CAP = 0.03
N_NORMAL = 50


def load_tags(tag_dir: str) -> dict:
    out = {}
    for path in sorted(glob.glob(f"{tag_dir}/rank*.jsonl")):
        with open(path) as handle:
            for line in handle:
                r = json.loads(line)
                if r.get("tags") is not None and len(r["tags"]) == len(r["sentences"]):
                    out[r["study_uid"]] = r
    return out


def abnormal(tag: dict) -> bool:
    return bool(tag["present"] or tag["possible"])


def join(sentences) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())


def insert_after_abnormal(sentences, tags, donor: str) -> str:
    last = max((i for i, t in enumerate(tags) if abnormal(t)), default=-1)
    return join(list(sentences[:last + 1]) + [donor] + list(sentences[last + 1:]))


def build_pairs(tags: dict, patient_of: dict, positives: dict, planned: dict, classes: list[str], seed: int = 0):
    rng = random.Random(seed)
    donors = defaultdict(list)                 # class -> [(patient, sentence)] asserting only that class
    normals = Counter()
    for uid, r in tags.items():
        for s, t in zip(r["sentences"], r["tags"]):
            if len(t["present"]) == 1 and not t["possible"]:
                donors[t["present"][0]].append((patient_of.get(uid), s))
            if not abnormal(t) and t["absent"] and len(s) < 160:
                normals[s] += 1
    normal_pool = [s for s, _ in normals.most_common(N_NORMAL)]
    prevalence = Counter(c for v in positives.values() for c in v)
    cap = {c: max(1, int(INSERT_CAP * len(tags))) for c in classes}
    used = Counter()
    pairs = []

    def donor_for(c, patient):
        pool = [s for p, s in donors.get(c, []) if p != patient]
        return rng.choice(pool) if pool else None

    for uid in sorted(tags):
        r = tags[uid]
        sents, tg = r["sentences"], r["tags"]
        if not sents:
            continue
        chosen_sha = r["sha256"]
        add = lambda kind, text, attr="": pairs.append((uid, kind, attr, chosen_sha, text))  # noqa: E731
        bad = [i for i, t in enumerate(tg) if abnormal(t)]
        stated = {c for t in tg for c in t["present"] + t["possible"]}
        positive = set(positives.get(uid, ()))
        patient = patient_of.get(uid)
        for i in rng.sample(bad, min(2, len(bad))):
            add("omission", join(s for k, s in enumerate(sents) if k != i))
        if len(bad) >= 3:
            drop = set(rng.sample(bad, rng.randint(2, min(4, len(bad) - 1))))
            add("multi_omission", join(s for k, s in enumerate(sents) if k not in drop))
        candidates = [c for c in planned.get(uid, ()) if c not in stated | positive | NO_INSERT and used[(c, "p")] < cap[c]]
        rng.shuffle(candidates)
        for c in candidates:
            donor = donor_for(c, patient)
            if donor:
                used[(c, "p")] += 1
                add("insertion_plan_fp", insert_after_abnormal(sents, tg, donor), c)
                break
        pool = [c for c in classes if c not in stated | positive | NO_INSERT and donors.get(c) and used[(c, "r")] < cap[c]]
        if pool:
            weights = [math.sqrt(prevalence.get(c, 0) + 1) for c in pool]
            c = rng.choices(pool, weights)[0]
            donor = donor_for(c, patient)
            if donor:
                used[(c, "r")] += 1
                add("insertion_random", insert_after_abnormal(sents, tg, donor), c)
        if bad and normal_pool:
            normal = rng.choice(normal_pool)
            kept = [normal if k == bad[0] else s for k, s in enumerate(sents) if k == bad[0] or k not in bad]
            add("normalised", join(kept))
        if len(sents) >= 2 and rng.random() < 0.5:
            i = rng.randrange(len(sents))
            add("repetition", join(sents[:i + 1] + [sents[i]] + sents[i + 1:]))
        elif len(sents) >= 2:
            tail = sents[-rng.randint(1, 2):]
            add("repetition", join(sents + tail * rng.randint(2, 3)))
    return [p for p in pairs if p[4].strip()], dict(donor_classes=len(donors), normal_pool=len(normal_pool))


def main(argv=None):
    from .data import read_reports
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", required=True, help="output directory of tag_sentences")
    ap.add_argument("--reports", required=True, help="report CSV glob (the chosen texts are re-checked against it)")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--splits-csv", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    calibration = json.loads(Path(a.calibration).read_text())
    classes = list(calibration["labels"])
    tags = load_tags(a.tags)
    reports = read_reports(a.reports)
    stale = [u for u, r in tags.items() if sha256_text(reports.get(u, "")) != r["sha256"]]
    for u in stale:
        tags.pop(u)
    with open(a.splits_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    patient_of = {r["study_uid"]: r["patient_uid"] for r in rows}
    train = {r["study_uid"] for r in rows if r["split"] == "train"}
    tags = {u: r for u, r in tags.items() if u in train}
    positives = {}
    with open(a.labels_csv, newline="") as handle:
        for r in csv.DictReader(handle):
            positives[r["study_uid"]] = [c for c in classes if r.get(c, "").strip() in ("1", "1.0")]
    z = np.load(a.dense, allow_pickle=False)
    slope, intercept = np.array(calibration["class_slope"]), np.array(calibration["class_intercept"])
    possible = np.array(calibration["class_possible_threshold"])
    p = 1 / (1 + np.exp(-(z["logit"] * slope + intercept)))
    planned = {str(u): [classes[j] for j in np.flatnonzero(row >= possible)] for u, row in zip(z["report_uid"], p)}
    pairs, info = build_pairs(tags, patient_of, positives, planned, classes, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE pairs(pair_id INTEGER PRIMARY KEY, report_uid TEXT, pair_type TEXT, attribute_type TEXT, "
                "chosen_sha256 TEXT, rejected_text TEXT)")
    con.executemany("INSERT INTO pairs(report_uid, pair_type, attribute_type, chosen_sha256, rejected_text) VALUES (?,?,?,?,?)",
                    pairs)
    con.execute("CREATE INDEX pairs_uid ON pairs(report_uid)")
    summary = dict(schema=SCHEMA, studies=len(tags), stale_tags=len(stale), pairs=len(pairs),
                   pairs_by_type=dict(Counter(p[1] for p in pairs)), paired_studies=len({p[0] for p in pairs}), **info)
    con.execute("INSERT INTO metadata VALUES ('summary', ?)", (json.dumps(summary),))
    con.commit()
    con.close()
    tmp.replace(out)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
