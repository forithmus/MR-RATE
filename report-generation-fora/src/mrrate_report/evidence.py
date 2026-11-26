"""Per-finding evidence of the MR report writer: description and focus tokens.

For a planned finding c the positive evidence e[t, c] of every token of every sequence (``mil.MRVisualStream``) is
summed per atlas token cell (all sequences share the atlas grid), smoothed with a 4 mm Gaussian, and cut to the
smallest set of cells holding 50 % of the smoothed mass. Its 26-connected components are the foci, ordered by mass
(at most 8).

* Description: distribution (focal, multifocal with the number of sites, or diffuse), the main site and up to two
  further sites as brain-region phrases (``regions.describe``), and the sequences that hold most of the evidence.
  Whole-brain and device findings get no site.
* Focus tokens: 16 tokens spread over the foci (at least one per focus, the rest by mass), chosen inside each focus by
  farthest-point sampling from its peak. Each is the classifier input x of the sequence token with the most evidence
  in that cell, followed by 16 auxiliary features (``AUX_FIELDS``). The attention-pooled finding feature comes first
  with zero auxiliary features.

One token cell is (2, 16, 16) voxels of the classifier input = 2 x 8 x 8 mm.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label

from .regions import describe

N_FOCI, N_TOKENS = 8, 16
AUX_FIELDS = ("evidence", "focus_rank", "focus_share", "z", "y", "x", "dz", "dy", "dx", "is_peak", "n_foci",
              "extent_z", "extent_y", "extent_x", "sequence_kind", "smoothed")
AUX_DIM = len(AUX_FIELDS)
# diagnoses whose evidence is global or a device: no site in the description
NO_SITE = {"cerebral_atrophy", "small_vessel_ischemic_disease", "ventriculomegaly", "hydrocephalus_ex_vacuo",
           "obstructive_hydrocephalus", "intracranial_hypotension", "multiple_sclerosis", "neuromyelitis_optica",
           "craniotomy_craniectomy", "postsurgical_changes", "catheter", "spinal_degenerative_changes"}
SEQUENCE_NAMES = {"t1": "T1", "t1ce": "contrast-enhanced T1", "t2": "T2", "flair": "FLAIR", "swi": "SWI",
                  "t2star": "T2*", "dwi": "DWI", "adc": "ADC", "pd": "PD", "stir": "STIR", "tof": "TOF angiography",
                  "other": "other"}
DIFFUSE_FRACTION = 0.15     # HDR50 cells / brain cells above which the evidence is called diffuse


class EvidenceGeometry:
    """Token grid, cell sizes and the brain-region id of every cell (``regions.build_region_map``)."""

    def __init__(self, grid, patch, spacing, region_map):
        self.grid = tuple(int(g) for g in grid)
        self.patch = np.asarray(patch, np.int64)
        self.spacing = np.asarray(spacing, np.float64)
        self.cell_mm = self.patch * self.spacing
        if tuple(region_map.grid) != self.grid:
            raise ValueError(f"region grid {region_map.grid} != token grid {self.grid}")
        self.region = region_map.token_region.astype(np.int64)
        self.keys = region_map.keys
        self.brain = np.array([not k[2] for k in self.keys])[self.region]
        self.n_regions = len(self.keys)


def _smooth(cells: np.ndarray, geometry: EvidenceGeometry) -> np.ndarray:
    return gaussian_filter(cells.reshape(geometry.grid), sigma=4.0 / geometry.cell_mm, mode="constant")


def foci(cell_evidence: np.ndarray, geometry: EvidenceGeometry):
    """Smoothed map, HDR50 mask and up to ``N_FOCI`` foci (dicts with cells, mass share, centroid, peak, extent)."""
    sm = _smooth(cell_evidence, geometry)
    total = float(sm.sum())
    if total <= 0:
        return sm, np.zeros_like(sm, bool), []
    order = np.sort(sm.ravel())[::-1]
    cut = order[min(int(np.searchsorted(np.cumsum(order), 0.5 * total)), len(order) - 1)]
    mask = (sm >= cut) & (sm > 0)
    lab, n = label(mask, structure=np.ones((3, 3, 3), bool))
    found = []
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        weights = sm[tuple(cells.T)]
        found.append(dict(cells=cells, weights=weights, mass=float(weights.sum())))
    found.sort(key=lambda f: -f["mass"])
    hdr_mass = sum(f["mass"] for f in found)
    for rank, f in enumerate(found[:N_FOCI]):
        f["rank"] = rank
        f["share"] = f["mass"] / hdr_mass
        f["centroid"] = (f["cells"] * f["weights"][:, None]).sum(0) / f["weights"].sum()
        f["peak"] = f["cells"][int(np.argmax(f["weights"]))]
        f["extent_mm"] = (f["cells"].max(0) - f["cells"].min(0) + 1) * geometry.cell_mm
    return sm, mask, found[:N_FOCI]


def _region_masses(cells: np.ndarray, values: np.ndarray, geometry: EvidenceGeometry) -> np.ndarray:
    flat = np.ravel_multi_index(tuple(cells.T), geometry.grid)
    return np.bincount(geometry.region[flat], weights=values, minlength=geometry.n_regions)


def describe_finding(name: str, cell_evidence: np.ndarray, sm: np.ndarray, mask: np.ndarray, found: list,
                     geometry: EvidenceGeometry, sequence_share: dict) -> str:
    """Distribution, main and further sites, and the sequences holding the evidence."""
    parts = []
    if found and name not in NO_SITE:
        brain_cells = max(1, int(geometry.brain.sum()))
        if int(mask.sum()) >= DIFFUSE_FRACTION * brain_cells:
            distribution = "diffuse"
        elif len(found) == 1:
            distribution = "focal"
        else:
            distribution = f"multifocal, {len(found)} sites"
        sites = []
        for f in found[:3]:
            phrase = describe(_region_masses(f["cells"], cell_evidence[np.ravel_multi_index(tuple(f["cells"].T),
                                                                                           geometry.grid)],
                                             geometry), geometry.keys)
            if phrase and phrase not in sites:
                sites.append(phrase)
        whole = describe(np.bincount(geometry.region, weights=cell_evidence, minlength=geometry.n_regions),
                         geometry.keys)
        if "both" in whole or "bilateral" in whole:
            distribution += ", bilateral"
        parts.append(distribution)
        if sites:
            parts.append(f"main site {sites[0]}")
        if len(sites) > 1:
            parts.append("further sites " + " and ".join(sites[1:]))
    leading = [k for k, v in sorted(sequence_share.items(), key=lambda kv: -kv[1]) if v >= 0.25][:2]
    if leading:
        parts.append("evidence mainly on " + " and ".join(SEQUENCE_NAMES.get(k, k) for k in leading))
    return "; ".join(parts)


def _farthest_points(cells: np.ndarray, start: int, k: int, cell_mm: np.ndarray) -> list[int]:
    pts = cells * cell_mm
    chosen = [int(start)]
    dist = np.linalg.norm(pts - pts[start], axis=1)
    while len(chosen) < min(k, len(cells)):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return chosen


def allocate(shares, total: int = N_TOKENS) -> list[int]:
    """At least one token per focus, the rest by mass share (largest remainder)."""
    n = min(len(shares), total)
    if n == 0:
        return []
    counts = [1] * n
    rest = total - n
    want = np.asarray(shares[:n], np.float64) * rest / max(sum(shares[:n]), 1e-12)
    extra = np.floor(want).astype(int)
    for i in np.argsort(-(want - extra))[: rest - int(extra.sum())]:
        extra[i] += 1
    return [c + int(e) for c, e in zip(counts, extra)]


def focus_tokens(enc, c: int, sm: np.ndarray, found: list, geometry: EvidenceGeometry, kind_count: int):
    """[k, D + AUX_DIM] focus tokens of finding c (k <= 16); empty without foci."""
    if not found:
        return torch.zeros(0, enc.tokens.shape[1] + AUX_DIM, device=enc.tokens.device)
    e = enc.evidence[:, c]
    e_max = float(e.max().clamp_min(1e-12))
    sm_max = float(sm.max()) or 1.0
    grid = np.asarray(geometry.grid, np.float64)
    rows, aux = [], []
    for f, count in zip(found, allocate([f["share"] for f in found])):
        start = int(np.argmax(f["weights"]))
        for j, idx in enumerate(_farthest_points(f["cells"], start, count, geometry.cell_mm)):
            cell = f["cells"][idx]
            flat = int(np.ravel_multi_index(tuple(cell), geometry.grid))
            members = torch.nonzero(enc.cell == flat).flatten()
            if members.numel() == 0:
                continue
            t = members[int(torch.argmax(e[members]))]
            offset = (cell - f["centroid"]) * geometry.cell_mm / 50.0
            aux.append([float(e[t]) / e_max, f["rank"] / N_FOCI, f["share"], *(cell / grid), *offset,
                        float(j == 0), len(found) / N_FOCI, *(f["extent_mm"] / 50.0),
                        float(enc.view_kind[t]) / max(kind_count - 1, 1), float(sm[tuple(cell)]) / sm_max])
            rows.append(enc.tokens[t].float())
    if not rows:
        return torch.zeros(0, enc.tokens.shape[1] + AUX_DIM, device=enc.tokens.device)
    return torch.cat((torch.stack(rows), torch.tensor(aux, dtype=torch.float32, device=enc.tokens.device)), 1)


# ------------------------------------------------------------------------------------------------ one finding
def finding_evidence(enc, c: int, name: str, geometry: EvidenceGeometry) -> dict:
    """Description and tokens [1 + k, D + AUX_DIM] (pooled first) of planned finding ``c``."""
    e = enc.evidence[:, c].double()
    n_cells = int(np.prod(geometry.grid))
    cells = torch.zeros(n_cells, dtype=torch.float64, device=e.device).index_add_(0, enc.cell, e).cpu().numpy()
    per_sequence = torch.zeros(len(enc.kinds), dtype=torch.float64, device=e.device).index_add_(0, enc.sequence, e)
    total = float(per_sequence.sum())
    share: dict[str, float] = {}
    if total > 0:
        from mr_dino.geometry import SEQUENCE_KINDS
        for k, v in zip(enc.kinds, (per_sequence / total).tolist()):
            share[SEQUENCE_KINDS[k]] = share.get(SEQUENCE_KINDS[k], 0.0) + v
    sm, mask, found = foci(cells, geometry)
    pooled = torch.cat((enc.pooled[c].float(), torch.zeros(AUX_DIM, device=enc.pooled.device)))[None]
    tokens = torch.cat((pooled, focus_tokens(enc, c, sm, found, geometry, len(SEQUENCE_NAMES))))
    return dict(description=describe_finding(name, cells, sm, mask, found, geometry, share), tokens=tokens,
                n_foci=len(found))
