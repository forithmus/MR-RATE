"""CPU tests of the MR writer inputs: evidence foci and tokens, calibration tiers, plan text, sentence tags, DPO pairs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


from mrrate_report import calibrate, dpo_pairs, evidence  # noqa: E402
from mrrate_report.regions import build_region_map  # noqa: E402
from mrrate_report.schema import label_order_sha256  # noqa: E402
from mrrate_report.tag_sentences import parse, sha256_text, split_sentences  # noqa: E402

GRID, PATCH, SPACING = (128, 24, 24), (2, 16, 16), (1.0, 0.5, 0.5)


@pytest.fixture(scope="module")
def geometry():
    return evidence.EvidenceGeometry(GRID, PATCH, SPACING, build_region_map())


def fake_encoding(cells_and_values, n_classes=2, dim=8, kinds=(3, 4)):
    """Two sequences covering the same cells; class 0 evidence at the given cells, sequence 0 holds 3/4 of it."""
    cells, values = zip(*cells_and_values)
    cells = torch.tensor(cells * 2)
    seq = torch.tensor([0] * len(values) + [1] * len(values))
    e = torch.zeros(len(cells), n_classes)
    e[:, 0] = torch.tensor([v * .75 for v in values] + [v * .25 for v in values])
    tokens = torch.randn(len(cells), dim)
    return SimpleNamespace(tokens=tokens.bfloat16(), evidence=e, cell=cells, sequence=seq,
                           view_kind=torch.tensor(kinds)[seq], kinds=list(kinds), pooled=torch.randn(n_classes, dim))


def flat(z, y, x):
    return int(np.ravel_multi_index((z, y, x), GRID))


def test_two_foci_tokens_and_description(geometry):
    lesion_a = [(flat(80 + dz, 6 + dy, 6), 5.0) for dz in range(3) for dy in range(2)]     # one side, large
    lesion_b = [(flat(40, 18, 18 + dx), 12.0) for dx in range(2)]                           # far away, dense
    enc = fake_encoding(lesion_a + lesion_b)
    out = evidence.finding_evidence(enc, 0, "meningioma", geometry)
    assert out["n_foci"] == 2
    tokens = out["tokens"]
    assert tokens.shape[1] == 8 + evidence.AUX_DIM
    assert 3 <= tokens.shape[0] <= 1 + evidence.N_TOKENS
    assert torch.all(tokens[0, 8:] == 0)                      # pooled token first, zero auxiliary features
    assert torch.allclose(tokens[0, :8], enc.pooled[0])
    assert "multifocal, 2 sites" in out["description"] and "main site" in out["description"]
    assert "evidence mainly on FLAIR" in out["description"]   # kind 3 holds 75 %
    whole = evidence.finding_evidence(enc, 0, "cerebral_atrophy", geometry)
    assert "site" not in whole["description"] and "FLAIR" in whole["description"]
    empty = evidence.finding_evidence(enc, 1, "glioma", geometry)
    assert empty["n_foci"] == 0 and empty["tokens"].shape[0] == 1


def test_allocation_keeps_every_focus():
    counts = evidence.allocate([0.9, 0.05, 0.05])
    assert sum(counts) == 16 and min(counts) >= 1 and counts[0] == max(counts)
    assert sum(evidence.allocate([0.5, 0.3, 0.2])) == 16
    assert evidence.allocate([]) == []


def test_calibration_tiers_use_train_only_and_mask_unknown():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random((n, 3)) < [0.3, 0.1, 0.0]
    X = np.where(y, 1.5, -1.0) + rng.normal(0, 1, (n, 3))
    Y = y.astype(float)
    Y[:20, 0] = np.nan                                        # unknown labels are ignored
    splits = np.array(["train"] * 300 + ["val"] * 100)
    X[300:] += 50                                             # validation logits must not move the cutoffs
    c = calibrate.build(np.arange(n).astype(str), splits, X, Y, ["a", "b", "c"])
    c2 = calibrate.build(np.arange(n).astype(str), splits, np.r_[X[:300], X[300:] - 50], Y, ["a", "b", "c"])
    assert np.allclose(c["plan"], c2["plan"]) and np.allclose(c["slope"], c2["slope"])
    assert c["labelled"][0] == 280 and c["positives"][2] == 0
    assert c["plan"][2] > 1 and c["possible"][2] > 1          # never planned without train positives
    assert np.all(c["possible"][:2] <= c["plan"][:2]) and np.all(c["plan"][:2] >= 0.5 - 1e-9)


def test_plan_text_mr_wording(tmp_path):
    pytest.importorskip("fora")
    from fora.reportgen.plan import diagnostic_plan as DP
    from mrrate_report.plan import MRPlanBuilder
    labels = ["meningioma", "glioma", "cerebral_atrophy"]
    cal = dict(labels=labels, label_order_sha256=label_order_sha256(labels),
               canonical_plan_labels=["meningioma", "glioma", "cerebral atrophy"], class_slope=[1, 1, 1],
               class_intercept=[0, 0, 0], class_plan_threshold=[.5, .5, .5], class_possible_threshold=[.2, .2, .2],
               class_val_positive=[10, 5, 50], n_calibration_studies=100)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal))
    builder = MRPlanBuilder(path, labels, tokenizer=None)
    logits = torch.tensor([[2.0, -1.0, -3.0]])
    text = builder.plan_content(logits, {-1: "FLAIR, T1", 0: "focal; main site right cerebellopontine angle"}, {0: 5, 1: 1})
    assert "brain MRI" in text and "MRI sequences in this study: FLAIR, T1." in text
    assert "Confident findings: meningioma (calibrated probability 0.88" in text
    assert "main site right cerebellopontine angle)" in text
    assert text.count(DP.FINDING_PLACEHOLDER) == 5 + 1 and DP.FINDING_IMAGE_PLACEHOLDER not in text
    assert "Possible findings (lower-confidence classifier signal; verify carefully): glioma" in text
    assert builder.planned_finding_order(logits, {0: 5, 1: 1}) == [0, 1]


def test_sentences_and_tag_parsing():
    text = "A 3.5 mm lesion is seen in the right frontal lobe. No mass effect.\nVentricles are normal."
    sents = split_sentences(text)
    assert sents == ["A 3.5 mm lesion is seen in the right frontal lobe.", "No mass effect.", "Ventricles are normal."]
    raw = '```json\n{"sentences": [{"n": 1, "present": ["meningioma", "bogus"]}, {"n": 2, "absent": ["brain_mass_effect"]}]}\n```'
    tags = parse(raw, 3, {"meningioma", "brain_mass_effect"})
    assert tags[0]["present"] == ["meningioma"] and tags[1]["absent"] == ["brain_mass_effect"] and tags[2]["present"] == []
    assert parse("no json", 3, set()) is None


def test_pairs_follow_the_rules():
    def rec(sents, tags):
        text = " ".join(sents)
        return dict(sentences=sents, tags=[dict(present=p, possible=q, absent=a) for p, q, a in tags], sha256=sha256_text(text))
    tags = {
        "s1": rec(["Right frontal meningioma.", "Small left parietal infarct.", "Lacune in the pons.", "No hydrocephalus."],
                  [(["meningioma"], [], []), (["chronic_ischemic_stroke"], [], []), (["lacunar_stroke"], [], []),
                   ([], [], ["obstructive_hydrocephalus"])]),
        "s2": rec(["Glioma in the left temporal lobe.", "No hydrocephalus."],
                  [(["glioma"], [], []), ([], [], ["obstructive_hydrocephalus"])]),
    }
    classes = ["meningioma", "chronic_ischemic_stroke", "lacunar_stroke", "glioma", "obstructive_hydrocephalus"]
    pairs, _ = dpo_pairs.build_pairs(tags, {"s1": "p1", "s2": "p2"}, {"s1": ["meningioma"], "s2": ["glioma"]},
                                     {"s1": ["glioma"], "s2": ["meningioma", "lacunar_stroke"]}, classes)
    kinds = {(u, k) for u, k, *_ in pairs}
    assert ("s1", "omission") in kinds and ("s1", "multi_omission") in kinds and ("s1", "normalised") in kinds
    fp = [p for p in pairs if p[0] == "s1" and p[1] == "insertion_plan_fp"]
    assert fp and fp[0][2] == "glioma" and "Glioma in the left temporal lobe." in fp[0][4]
    for uid, kind, attr, sha, rejected in pairs:
        assert sha == tags[uid]["sha256"] and rejected != " ".join(tags[uid]["sentences"])
        if kind == "normalised":
            assert "No hydrocephalus." in rejected and "meningioma" not in rejected.lower()
