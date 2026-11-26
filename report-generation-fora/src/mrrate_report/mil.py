"""Frozen MR-RATE MIL (ViT-H+ DINO, 74 diagnoses) as the visual stream of the report writer.

Per study, every eligible sequence goes through exactly the classifier's feature path:
    sequence -> frozen prefix (all blocks but the last) -> fine-tuned last block -> x = [(y - mean) / rms,
    global-grid Fourier position features]   (1280 + 24 = 1304-d)
and the MIL head scores every token of every sequence:
    score s[t, c], token logit l[t, c], attention w[t, c] = softmax over all tokens of the study,
    study logit = scale_c * sum_t w l + bias_c,  attention-pooled finding feature f[c] = sum_t w[t, c] x[t],
    signed evidence e[t, c] = w l scale_c (its positive part is the evidence map).
The writer receives every token x (with its sequence kind and sequence ordinal) through its resampler, the pooled
feature of each planned finding, and the focus tokens and images built from the evidence map (``evidence``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MAX_SERIES = 32        # sequence ordinal embedding size of the writer


def add_bundle_paths(bundle: str | Path) -> Path:
    """Put the bundle's frozen ``mr_dino`` code and ``dfw_mil`` on the import path."""
    bundle = Path(bundle).resolve()
    for p in (bundle / "code/MR-RATE_8a55c42/dino-pretraining", bundle):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return bundle


class MRVisualStream:
    """MIL bundle directory -> per-study tokens, logits, pooled finding features and evidence."""

    def __init__(self, bundle: str | Path, device, checkpoint: str = "checkpoint/epoch_004.pt"):
        bundle = add_bundle_paths(bundle)
        from mr_dino.geometry import SEQUENCE_KINDS, position_features
        from mr_dino.mil_head import SparseClassifyThenAggregate
        from mr_dino.mil_stream import LastBlockAdapter
        from mr_dino.release import load_release
        self.device = torch.device(device)
        self._position_features = position_features
        self.n_kinds = len(SEQUENCE_KINDS)
        package = torch.load(bundle / checkpoint, map_location="cpu", weights_only=False)
        self.model, self.meta = load_release(bundle / "backbone_release", self.device)
        if self.meta["config"] != package["release_config"]:
            raise ValueError("MIL checkpoint belongs to another DINO release")
        self.adapter = LastBlockAdapter(self.model).to(self.device)
        self.adapter.load_state_dict(package["adapter"])
        self.adapter.eval().requires_grad_(False)
        arguments = package["arguments"]
        self.freqs, self.pscale = int(arguments["position_freqs"]), float(arguments["position_scale"])
        self.tile_batch = int(arguments["tile_batch"])
        self.classes = list(package["classes"])
        self.dim = self.adapter.dim + 6 * self.freqs
        self.head = SparseClassifyThenAggregate(self.dim, len(self.classes), use_norm=False,
                                                checkpoint_chunks=False).to(self.device)
        self.head.route_bias_raw = torch.nn.Parameter(torch.zeros(len(self.classes), device=self.device))
        self.head.load_state_dict(package["head"])
        self.head.eval().requires_grad_(False)
        self.mean = package["normalization"]["mean"].float().to(self.device)
        self.rms = float(package["normalization"]["rms"])
        args = self.meta["args"]
        self.target_shape = tuple(args["target_shape"])
        self.spacing = tuple(float(s) for s in args["target_spacing"])
        self.patch = tuple(self.meta["config"]["backbone"]["patch_size"])
        self.grid = tuple(s // p for s, p in zip(self.target_shape, self.patch))
        self.checkpoint = str((bundle / checkpoint).resolve())
        del package

    def prepare(self, study: dict):
        """Host side (loader threads): ``[(image, support)]`` per sequence, the classifier's preprocessing."""
        from dfw_mil.fast import load_study
        return load_study(study, self.meta)

    @torch.no_grad()
    def encode(self, study: dict, volumes) -> SimpleNamespace:
        from dfw_mil.fast import cached_study_factory
        from mr_dino.geometry import sequence_kind
        grid = torch.tensor(self.grid, device=self.device)
        xs, scores, logits_t, cells, seqs, kinds = [], [], [], [], [], []
        classes = torch.arange(len(self.classes), device=self.device)
        for s, (path, volume) in enumerate(zip(study["image_paths"], volumes)):
            one = {"study_uid": study["study_uid"], "image_paths": [path]}
            factory = cached_study_factory(one, [volume], self.model, self.adapter, self.meta, self.device,
                                           self.tile_batch)
            for y, _, coordinates, _ in factory():
                x = torch.cat(((y - self.mean) / self.rms,
                               self._position_features(coordinates, self.freqs) * self.pscale), 1)
                sc, lg = self.head._chunk_outputs(x.float(), classes)
                xs.append(x.to(torch.bfloat16))
                scores.append(sc.float())
                logits_t.append(lg.float())
                g = torch.round(coordinates * grid - .5).long()
                cells.append((g[:, 0] * grid[1] + g[:, 1]) * grid[2] + g[:, 2])
                seqs.append(torch.full((len(x),), s, dtype=torch.long, device=self.device))
            kinds.append(sequence_kind(path))
        tokens = torch.cat(xs)
        S, L = torch.cat(scores), torch.cat(logits_t)
        w = torch.softmax(S, 0)
        scale, bias = self.head.output_scale.float(), self.head.output_bias.float()
        logits = (w * L).sum(0) * scale + bias
        pooled = torch.zeros(len(self.classes), self.dim, device=self.device)
        for start in range(0, len(tokens), 65536):
            pooled += w[start:start + 65536].t() @ tokens[start:start + 65536].float()
        sequence = torch.cat(seqs)
        kind_table = torch.tensor(kinds, dtype=torch.long, device=self.device)
        return SimpleNamespace(
            tokens=tokens, view_kind=kind_table[sequence], view_series=sequence.clamp(max=MAX_SERIES - 1),
            sequence=sequence, cell=torch.cat(cells), logits=logits, pooled=pooled,
            evidence=(w * L * scale).clamp_min(0), total_tokens=int(len(tokens)), kinds=kinds,
            paths=list(study["image_paths"]))
