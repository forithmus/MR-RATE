"""Validation MIL probe: checkpoint loading, token extraction, patient-level CV, metrics."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mr_dino.probe import (
    TEACHER_PREFIX,
    TokenBags,
    auroc,
    average_precision,
    embed_sequence,
    export_teacher,
    load_teacher_backbone,
    main,
    patient_folds,
)
from mr_dino.train_ddp import build_backbone, check_checkpoint_space

TESTS = Path(__file__).resolve().parent


def write_dcp_checkpoint(path: Path, backbone, args: dict, step: int = 7) -> Path:
    import torch.distributed.checkpoint as dcp

    state = {TEACHER_PREFIX + k: v.clone() for k, v in backbone.state_dict().items()}
    state["model.student.backbone.cls_token"] = torch.zeros_like(backbone.cls_token)   # ignored by the loader
    dcp.save(state, storage_writer=dcp.FileSystemWriter(str(path)), no_dist=True)
    (path / "metadata.json").write_text(json.dumps({"step": step, "stage": "pretrain", "args": args}))
    (path / "COMPLETE").touch()
    return path


def tiny_args(**extra) -> dict:
    return {"arch": "tiny", "space": "coreg_space", "target_shape": [8, 32, 32],
            "target_spacing": [1.0, 0.5, 0.5], "posterior_shift_mm": 0.0, **extra}


def test_teacher_backbone_loads_from_dcp_and_export(tmp_path):
    torch.manual_seed(0)
    reference = build_backbone("tiny").eval()
    reference.init_weights()
    ckpt = write_dcp_checkpoint(tmp_path / "step_00000007", reference, tiny_args())
    loaded, meta = load_teacher_backbone(ckpt)
    assert meta["step"] == 7 and meta["args"]["arch"] == "tiny"
    x = torch.randn(1, 1, 8, 32, 32)
    expected = reference.forward_features(x)["x_norm_patchtokens"]
    torch.testing.assert_close(loaded.forward_features(x)["x_norm_patchtokens"], expected)
    exported = export_teacher(ckpt, tmp_path / "teacher.pt")
    again, meta2 = load_teacher_backbone(exported)
    assert meta2["step"] == 7
    torch.testing.assert_close(again.forward_features(x)["x_norm_patchtokens"], expected)


def test_embed_sequence_keeps_pooled_foreground_only():
    torch.manual_seed(0)
    backbone = build_backbone("tiny").eval()
    backbone.init_weights()
    volume = torch.full((16, 64, 64), -0.4)
    volume[2:10, 8:40, 8:40] = 0.3            # "head" occupies part of the volume
    tokens, coords = embed_sequence(backbone, volume, tile=(8, 32, 32), pool=(2, 2, 2))
    # token grid 8x4x4, pooled 4x2x2; head covers token z 1..4, y/x 0..2 -> pooled z 0..2, y/x 0..1
    assert tokens.dtype == torch.float16 and tokens.shape[1] == backbone.embed_dim
    assert {tuple(c) for c in coords.tolist()} == {(z, y, x) for z in range(3) for y in range(2) for x in range(2)}
    assert torch.isfinite(tokens.float()).all()
    empty, _ = embed_sequence(backbone, torch.full((8, 32, 32), -0.4), tile=(8, 32, 32), pool=(2, 2, 2))
    assert len(empty) == 0


def test_patient_folds_keep_patients_together():
    uids = [f"s{i}" for i in range(40)]
    patient_of = {u: f"p{i // 3}" for i, u in enumerate(uids)}
    folds = patient_folds(uids, patient_of, 5, seed=1)
    for p in set(patient_of.values()):
        assert len({f for u, f in zip(uids, folds) if patient_of[u] == p}) == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
    assert (folds == patient_folds(uids, patient_of, 5, seed=1)).all()


def test_auroc_and_ap_match_definitions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)   # ties on purpose
    pos, neg = s[y == 1], s[y == 0]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert auroc(y, s) == pytest.approx(brute)
    assert auroc(np.zeros(5), np.arange(5.0)) is None
    assert average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1])) == pytest.approx((1 + 2 / 3) / 2)


def write_token_shards(root: Path, n=60, dim=16, seed=0):
    """Bags whose label-0 positives carry a signal token direction; two worker shards."""
    rng = np.random.default_rng(seed)
    uids = [f"study{i:03d}" for i in range(n)]
    labels = rng.integers(0, 2, (n, 2))
    root.mkdir(parents=True)
    for rank in range(2):
        mine = list(range(rank, n, 2))
        chunks, starts, counts, total = [], [], [], 0
        for i in mine:
            bag = rng.normal(size=(int(rng.integers(5, 12)), dim)).astype(np.float16)
            if labels[i, 0]:
                bag[0, 0] += 6.0
            chunks.append(bag)
            starts.append(total)
            counts.append(len(bag))
            total += len(bag)
        (root / f"tokens_rank{rank:04d}.f16").write_bytes(np.concatenate(chunks).tobytes())
        np.savez(root / f"index_rank{rank:04d}.npz", format="mrdino_probe_tokens_v1", dim=dim,
                 token_file=f"tokens_rank{rank:04d}.f16", study_uid=np.array([uids[i] for i in mine]),
                 start=np.array(starts), count=np.array(counts), checkpoint_step=7, checkpoint_stage="pretrain",
                 space="coreg_space", tile=np.array([8, 32, 32]), pool=np.array([2, 2, 2]))
    with open(root.parent / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_uid", "signal", "noise"])
        w.writerows([u, *labels[i]] for i, u in enumerate(uids))
    with open(root.parent / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_uid", "patient_uid", "split"])
        w.writerows([u, f"p{i // 2}", "val"] for i, u in enumerate(uids))
    return uids


def test_cv_finds_signal_and_predicts_every_study_once(tmp_path):
    uids = write_token_shards(tmp_path / "feat")
    bags = TokenBags(tmp_path / "feat")
    assert sorted(bags.uids) == uids
    common = ["--features-dir", str(tmp_path / "feat"), "--labels-csv", str(tmp_path / "labels.csv"),
              "--splits-csv", str(tmp_path / "splits.csv"), "--device", "cpu"]
    for fold in range(3):
        main(["cv", *common, "--out-dir", str(tmp_path / "out"), "--fold", str(fold), "--folds", "3",
              "--epochs", "15", "--batch-size", "4", "--hidden-dim", "16", "--mlp-hidden-dim", "16"])
    main(["summarize", *common, "--out-dir", str(tmp_path / "out"), "--folds", "3"])
    result = json.loads((tmp_path / "out" / "results.json").read_text())
    assert result["studies"] == 60 and result["checkpoint_step"] == 7
    signal = next(r for r in result["per_class"] if r["label"] == "signal")
    assert signal["auroc"] > 0.9


def test_extract_end_to_end_on_raw_coreg_tree(tmp_path):
    pytest.importorskip("nibabel")
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(TESTS / "make_dummy_aligned_dataset.py"), "--out", str(data),
                    "--studies", "5", "--shape", "8", "32", "32"], check=True)
    uids = sorted(p.name for p in (data / "batch00").iterdir())
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_uid", "a"])
        w.writerows([u, i % 2] for i, u in enumerate(uids))
    with open(tmp_path / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_uid", "patient_uid", "split"])
        w.writerows([u, u, "val" if i < 4 else "train"] for i, u in enumerate(uids))
    torch.manual_seed(0)
    backbone = build_backbone("tiny").eval()
    backbone.init_weights()
    ckpt = write_dcp_checkpoint(tmp_path / "ckpt", backbone, tiny_args())
    args = ["extract", "--checkpoint", str(ckpt), "--data-folder", str(data), "--features-dir", str(tmp_path / "feat"),
            "--labels-csv", str(tmp_path / "labels.csv"), "--splits-csv", str(tmp_path / "splits.csv"),
            "--tile", "8", "32", "32", "--workers", "0", "--device", "cpu"]
    main(args)
    bags = TokenBags(tmp_path / "feat")
    assert bags.uids == uids[:4]                    # val only, sorted
    assert all(len(bags.tokens(i)) > 0 for i in range(len(bags)))
    with np.load(tmp_path / "feat" / "index_rank0000.npz") as index:
        assert set(index["seq_id"].tolist()) == {0, 1, 2}      # every sequence contributes
    with pytest.raises(ValueError, match="trained on 'coreg_space'"):
        main([*args[:-2], "--device", "cpu", "--space", "atlas_space", "--features-dir", str(tmp_path / "f2")])


def test_checkpoint_space_guard():
    check_checkpoint_space("mrrate_aligned_dinov3d_fsdp2_v1", {"space": "coreg_space"}, "coreg_space",
                           "mrrate_aligned_dinov3d_fsdp2_v1")
    with pytest.raises(RuntimeError, match="trained on 'atlas_space'"):   # pre-coreg checkpoints are atlas
        check_checkpoint_space("mrrate_atlas_dinov3d_fsdp2_v1", {}, "coreg_space", "mrrate_aligned_dinov3d_fsdp2_v1")
    with pytest.raises(RuntimeError, match="not an MR DINO checkpoint"):
        check_checkpoint_space("something_else", {}, "coreg_space", "mrrate_aligned_dinov3d_fsdp2_v1")
