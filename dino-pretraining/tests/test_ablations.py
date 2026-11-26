"""MIL ablations: nested training subsets, cross-sequence attention, modality-channel inputs."""
import copy
import json
import random
from collections import Counter

import numpy as np
import pytest
import torch
from torch import nn

from dfw_mil.fast import cached_study_factory, cross_sequence_factory, load_study
from dfw_mil.fusion import CrossSequenceAttention
from dfw_mil.subset import subset_train_ids, train_patient_subset
from mr_dino.channels import (
    MODALITIES,
    acquisition_stacks,
    drop_channels,
    modality_groups,
    n_input_channels,
    random_subset,
    stack_channels,
)
from mr_dino.data import collate_dino3d
from mr_dino.data_channels import MRMultiChannelDINO3DDataset
from mr_dino.data import CropSpec
from mr_dino.geometry import SEQUENCE_KINDS
from mr_dino.mil_head import SparseClassifyThenAggregate
from mr_dino.mil_stream import LastBlockAdapter, backward
from mr_dino.model import DinoVisionTransformer3D
from mr_dino.train_ddp import build_backbone

NAMES = ["t1w-raw-axi.nii.gz", "t1w-raw-sag.nii.gz", "t2w-raw-axi.nii.gz", "flair-raw-axi.nii.gz",
         "swi-raw-axi.nii.gz", "mra-raw-axi.nii.gz"]


# ----------------------------------------------------------------------------- data fraction

def test_train_subsets_are_nested_patient_level_and_exact():
    patient_of = {f"s{i}": f"p{i // 3}" for i in range(300)}   # 100 patients, 3 studies each
    ids = list(patient_of)
    previous = set()
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        chosen = set(subset_train_ids(ids, patient_of, fraction, seed=7))
        patients = {patient_of[u] for u in chosen}
        assert len(patients) == round(100 * fraction)
        assert all(u in chosen for u in ids if patient_of[u] in patients)   # whole patients
        assert previous <= chosen
        previous = chosen
    assert train_patient_subset({f"p{i}" for i in range(100)}, 0.2, 7) != train_patient_subset(
        {f"p{i}" for i in range(100)}, 0.2, 8)
    with pytest.raises(ValueError):
        train_patient_subset({"p"}, 0.0, 0)


# ----------------------------------------------------------------------------- cross-sequence attention

def test_cross_sequence_attention_starts_as_identity_and_masks_missing_sequences():
    torch.manual_seed(0)
    module = CrossSequenceAttention(16, depth=2, heads=4)
    x = torch.randn(5, 3, 16)
    valid = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    kinds = torch.tensor([0, 2, 3])
    torch.testing.assert_close(module(x, valid, kinds), x)
    for p in module.parameters():
        nn.init.normal_(p, std=0.2)
    y = module(x, valid, kinds)
    changed = x.clone()
    changed[~valid] = 100.0   # values of missing sequences never reach covered ones
    torch.testing.assert_close(module(changed, valid, kinds)[valid], y[valid])
    order = torch.tensor([2, 0, 1])   # equivariant to the order of the sequences
    torch.testing.assert_close(module(x[:, order], valid[:, order], kinds[order]), y[:, order])
    with pytest.raises(ValueError):
        module(x, torch.zeros_like(valid), kinds)


def test_replay_gradients_with_cross_sequence_attention_match_dense_bag():
    torch.manual_seed(1)
    d, classes = 16, 3
    head = SparseClassifyThenAggregate(d, classes, hidden_dim=8, mlp_hidden_dim=7, use_norm=False, checkpoint_chunks=False)
    head.route_bias_raw = nn.Parameter(torch.tensor([1., 2., 3.]))
    fusion = CrossSequenceAttention(d, depth=1, heads=4)
    for p in fusion.parameters():
        nn.init.normal_(p, std=0.1)
    ref_head, ref_fusion = copy.deepcopy(head), copy.deepcopy(fusion)
    parts = [(torch.randn(6, 3, d), torch.rand(6, 3) > .3), (torch.randn(4, 2, d), torch.rand(4, 2) > .3)]
    parts = [(x, v | (~v.any(1, keepdim=True) & torch.tensor([True] + [False] * (v.shape[1] - 1)))) for x, v in parts]
    kinds = [torch.tensor([0, 2, 3]), torch.tensor([4, 0])]

    def factory():
        for (x, v), k in zip(parts, kinds):
            yield fusion(x, v, k)[v], None

    target = torch.tensor([1., 0., 1.])
    loss, logits = backward(head, factory, target)
    tokens = torch.cat([ref_fusion(x, v, k)[v] for (x, v), k in zip(parts, kinds)])
    scores, patch = ref_head._chunk_outputs(tokens, torch.arange(classes))
    expected = (scores.softmax(0) * patch).sum(0) * ref_head.output_scale + ref_head.output_bias
    nn.functional.binary_cross_entropy_with_logits(expected, target).backward()
    torch.testing.assert_close(logits[0], expected, atol=1e-6, rtol=1e-5)
    for actual, ref in zip(fusion.parameters(), ref_fusion.parameters()):
        torch.testing.assert_close(actual.grad, ref.grad, atol=2e-6, rtol=2e-5)
    for actual, ref in zip(head.parameters(), ref_head.parameters()):
        torch.testing.assert_close(actual.grad, ref.grad, atol=2e-6, rtol=2e-5)


# ----------------------------------------------------------------------------- modality channels

def test_modality_groups_stacks_and_planes():
    groups = modality_groups(NAMES)
    assert groups == [[0, 1], [2], [3], [4]]   # MRA is not an input modality
    assert acquisition_stacks(groups) == [[0, 2, 3, 4], [1, 2, 3, 4]]
    assert acquisition_stacks([[0, 1, 2], [], [3], []]) == [[0, None, 3, None], [1, None, 3, None], [2, None, 3, None]]
    volume = torch.full((2, 3, 4), 0.5)
    stack = stack_channels([volume, None, -volume, None])
    assert stack.shape == (n_input_channels(), 2, 3, 4)
    assert torch.equal(stack[4:].flatten(1).amax(1), torch.tensor([1., 0., 1., 0.]))
    assert torch.equal(stack[1], torch.zeros(2, 3, 4))
    dropped = drop_channels(stack, keep=(2,))
    assert torch.equal(dropped[4:].flatten(1).amax(1), torch.tensor([0., 0., 1., 0.]))
    assert torch.equal(dropped[0], torch.zeros(2, 3, 4)) and torch.equal(dropped[2], stack[2])
    rng = random.Random(0)
    subsets = Counter(random_subset([0, 2, 3], rng, full_probability=0.5) for _ in range(4000))
    assert subsets[(0, 2, 3)] > 2000 and all(s and set(s) <= {0, 2, 3} for s in subsets)


def test_multichannel_backbone_builds_with_availability_planes():
    model = build_backbone("tiny", n_input_channels()).eval()
    assert model.patch_embed.proj.weight.shape[1] == 2 * len(MODALITIES)
    out = model.forward_features(torch.randn(1, 8, 4, 32, 32))
    assert out["x_norm_patchtokens"].shape[1] == 2 * 2 * 2


def write_study(root, uid, names, shape=(8, 32, 32), seed=0):
    nib = pytest.importorskip("nibabel")
    folder = root / "batch00" / uid / "atlas_img"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    d, h, w = shape
    for i, name in enumerate(names):
        array = rng.normal(size=(h, w, d)).astype(np.float32) + i
        array[:3] = 0   # some air, so supports differ by sequence
        nib.save(nib.Nifti1Image(array, np.diag([0.5, 0.5, 1.0, 1.0])), folder / name)


def test_multichannel_dino_views(tmp_path):
    write_study(tmp_path, "study0", NAMES)
    write_study(tmp_path, "study1", NAMES[:1] + NAMES[3:4], seed=1)
    dataset = MRMultiChannelDINO3DDataset(
        data_folder=str(tmp_path), space="atlas_space",
        crop_spec=CropSpec(global_shape=(4, 32, 32), local_shape=(4, 16, 16), local_crops=2),
        target_shape=(8, 32, 32), target_spacing=(1.0, 0.5, 0.5), posterior_shift_mm=0,
        cross_sequence_probability=1.0, full_probability=0.0, patch_size=(2, 16, 16))
    assert len(dataset) == 2 and dataset.in_chans == 8
    for epoch in range(4):
        item = dataset[(0, epoch)]
        assert item["teacher_global"].shape == (2, 8, 4, 32, 32)
        assert item["student_local"].shape == (2, 8, 4, 16, 16)
        for view, code in zip(item["teacher_global"], item["global_sequences"]):
            kept = view[4:].flatten(1).amax(1) > 0
            assert int(sum(1 << c for c in range(4) if kept[c])) == code
            assert torch.equal(view[:4][~kept].abs().sum(), torch.tensor(0, dtype=view.dtype))
        assert torch.equal(dataset[(0, epoch)]["teacher_global"], item["teacher_global"])
    other = dataset[(1, 0)]
    assert set(other["global_sequences"]) <= {1, 4, 5}   # only T1 and FLAIR exist
    batch = collate_dino3d([dataset[(0, 0)], dataset[(1, 0)]], patch_size=(2, 16, 16))
    assert batch["teacher_global"].shape == (2, 2, 8, 4, 32, 32)


# ----------------------------------------------------------------------------- MIL inputs

def tiny_release(in_chans, input_channels=None):
    torch.manual_seed(4)
    model = DinoVisionTransformer3D(embed_dim=32, depth=2, num_heads=4, volume_size=(4, 32, 32), patch_size=(2, 16, 16),
                                    in_chans=in_chans, n_storage_tokens=2, drop_path_rate=0.)
    model.init_weights()
    model.eval().requires_grad_(False)
    config = {"training_crops": {"global": [4, 32, 32]}, "backbone": {"patch_size": [2, 16, 16], "in_chans": in_chans}}
    if input_channels:
        config["input_channels"] = input_channels
    meta = {"config": config, "args": {"target_shape": [8, 32, 32], "target_spacing": [1., .5, .5], "posterior_shift_mm": 0}}
    return model, LastBlockAdapter(model), meta


def test_cross_sequence_inputs_regroup_the_per_sequence_bag(tmp_path):
    write_study(tmp_path, "study0", NAMES[:5])
    study = {"study_uid": "study0", "image_paths": sorted(str(p) for p in (tmp_path / "batch00/study0/atlas_img").iterdir())}
    model, adapter, meta = tiny_release(1)
    volumes = load_study(study, meta)
    device = torch.device("cpu")
    plain = list(cached_study_factory(study, volumes, model, adapter, meta, device, 2)())
    grouped = list(cross_sequence_factory(study, volumes, model, adapter, meta, device, 2)())
    tokens = torch.cat([t for t, _, _, _ in plain])
    regrouped = torch.cat([t[v] for t, v, _, _ in grouped])
    assert regrouped.shape == tokens.shape
    key = lambda x: sorted(map(tuple, x.round(decimals=4).tolist()))
    assert key(regrouped) == key(tokens)
    fusion = CrossSequenceAttention(32, heads=4)
    fused = torch.cat([fusion(t, v, k)[v] for t, v, k, _ in grouped])
    assert key(fused) == key(tokens)   # untrained fusion = the per-sequence baseline bag
    t, v, k, c = grouped[0]
    assert [SEQUENCE_KINDS[i] for i in k.tolist()] == ["flair", "swi", "t1", "t1", "t2"]
    assert c.shape == (len(t), 3)


def test_modality_release_reads_every_acquisition_once_per_stack(tmp_path):
    write_study(tmp_path, "study0", NAMES)
    study = {"study_uid": "study0", "image_paths": sorted(str(p) for p in (tmp_path / "batch00/study0/atlas_img").iterdir())}
    layout = {"layout": "modality", "modalities": list(MODALITIES), "availability_planes": True}
    model, adapter, meta = tiny_release(8, layout)
    volumes = load_study(study, meta)
    assert sum(v is None for v in volumes) == 1   # the MRA is not read
    tiles = list(cached_study_factory(study, volumes, model, adapter, meta, torch.device("cpu"), 2)())
    assert len(tiles) == 2 * 2   # two T1 acquisitions -> two stacks, two tiles of depth 4 each
    assert all(t.shape[1] == 32 for t, _, _, _ in tiles)
    single, single_adapter, single_meta = tiny_release(1, layout)   # layout and channel count disagree
    with pytest.raises(ValueError, match="takes 1 channels"):
        list(cached_study_factory(study, volumes, single, single_adapter, single_meta, torch.device("cpu"))())


# ----------------------------------------------------------------------------- release export

def test_export_release_round_trip(tmp_path):
    import torch.distributed.checkpoint as dcp

    from mr_dino.export_release import export_release
    from mr_dino.release import release_metadata

    backbone = build_backbone("tiny", n_input_channels())
    backbone.init_weights()
    checkpoint = tmp_path / "checkpoints" / "step_00000010"
    dcp.save({f"model.teacher.backbone.{k}": v for k, v in backbone.state_dict().items()},
             storage_writer=dcp.FileSystemWriter(str(checkpoint)), no_dist=True)
    args = {"arch": "tiny", "space": "atlas_space", "stage": "pretrain", "target_shape": [256, 384, 384],
            "target_spacing": [1.0, 0.5, 0.5], "posterior_shift_mm": 15.0, "input_channels": "modality",
            "global_shape": None, "local_shape": None}
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 10, "stage": "pretrain", "args": args}))
    (checkpoint / "COMPLETE").touch()
    release = export_release(checkpoint, tmp_path / "release")
    meta = release_metadata(release)   # the tiny test architecture has no release constructor
    assert meta["config"]["backbone"]["in_chans"] == 8
    assert meta["config"]["input_channels"]["modalities"] == list(MODALITIES)
    assert meta["config"]["training_crops"]["global"] == [64, 192, 192]
    rebuilt = build_backbone("tiny", meta["config"]["backbone"]["in_chans"])
    rebuilt.load_state_dict(torch.load(release / meta["config"]["weights"], weights_only=True), strict=True)
    for k, v in backbone.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[k].float(), v.to(torch.bfloat16).float())
    with pytest.raises(FileExistsError):
        export_release(checkpoint, release)
