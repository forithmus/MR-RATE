"""Brain regions on the MIL token grid, and plain-language location phrases from evidence.

The region map is the NV-Segment-CTMR MRI_BRAIN parcellation in MR-RATE atlas (MNI) space, majority-voted over
24 train studies (assets/, built by scripts/build_region_atlas.py). It is resampled with nearest neighbour through
exactly the image pipeline of the MIL (canonical RAS, (Z,X,Y) transpose, resize to the target spacing,
crop/pad with the posterior shift), then every MIL token (2 x 8 x 8 mm) gets one region id:

* intra-axial: the most frequent region group inside the token footprint;
* cerebral white matter is named after the nearest cortical lobe ("right frontal white matter"), and white
  matter within 6 mm of a lateral ventricle is "periventricular white matter";
* tokens that are not mostly brain are "extra-axial" and named after the nearest brain group within 20 mm,
  everything further out is "outside the brain".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

ASSETS = Path(__file__).resolve().parents[2] / "assets"

# NV-Segment label name (side stripped, lower case) -> region group
GROUP_PATTERNS = [
    ("frontal lobe", r"superior-frontal|middle-frontal|frontal-pole|medial-frontal|gyrus-rectus|orbital-gyrus|"
                     r"part-of-the-ifg|frontal-operculum|precentral|supplementary-motor|subcallosal"),
    ("occipital lobe", r"calcarine|(?<!pre)cuneus|lingual|occipital"),
    ("parietal lobe", r"postcentral|superior-parietal|supramarginal|angular|precuneus|parietal-operculum"),
    ("temporal lobe", r"superior-temporal|middle-temporal|inferior-temporal|temporal-pole|planum-polare|"
                      r"planum-temporale|transverse-temporal|fusiform-gyrus$|parahippocampal|entorhinal"),
    ("insula", r"insula|central-operculum"),
    ("cingulate gyrus", r"cingulate"),
    ("hippocampus and amygdala", r"hippocampus|amygdala"),
    ("basal ganglia", r"caudate|putamen|pallidum|accumbens|basal-forebrain"),
    ("thalamus", r"thalamus"),
    ("hypothalamic and suprasellar region", r"ventral-dc"),
    ("cerebral white matter", r"cerebral-white-matter"),
    ("lateral ventricle", r"lateral-ventricle|inf-lat-vent"),
    ("third ventricle", r"3rd-ventricle"),
    ("fourth ventricle", r"4th-ventricle"),
    ("brainstem", r"brain-stem"),
    ("cerebellar hemisphere", r"cerebellum-exterior|cerebellum-white-matter"),
    ("cerebellar vermis", r"vermal"),
]
LOBES = ("frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe", "insula")
MIDLINE = {"third ventricle", "fourth ventricle", "brainstem", "cerebellar vermis"}


def _group_of(name: str) -> tuple[str, str]:
    side = "left" if name.startswith("Left-") else "right" if name.startswith("Right-") else "midline"
    bare = re.sub(r"^(Left|Right)-", "", name).lower()
    # the "-MPoG-postcentral" style names keep a gyrus code before the long name; match on the long name
    for group, pat in GROUP_PATTERNS:
        if re.search(pat, bare):
            return group, ("midline" if group in MIDLINE else side)
    raise KeyError(f"NV-Segment label {name!r} has no region group")


@dataclass
class RegionMap:
    token_region: np.ndarray      # (n_tokens,) int, index into keys; flat over the MIL global token grid
    keys: list[tuple[str, str, bool]]   # (group, side, extra_axial); side in left | right | midline | none
    grid: tuple[int, int, int]

    @property
    def names(self) -> list[str]:
        return [phrase(*k) for k in self.keys]


def _plural(group: str) -> str:
    for a, b in (("lobe", "lobes"), ("hemisphere", "hemispheres"), ("thalamus", "thalami"), ("ventricle", "ventricles"),
                 ("angle", "angles"), ("gyrus", "gyri"), ("insula", "insulae")):
        if group.endswith(a): return group[: -len(a)] + b
    return group


def phrase(group: str, side: str, extra: bool, bilateral: bool = False) -> str:
    both = bilateral or side == "both"
    def place(g):
        if g in ("midline",) or side in ("midline", "none"): return g
        if both:
            pl = _plural(g)
            return f"both {pl}" if pl != g else f"{g} on both sides"
        return f"{side} {g}"
    if group == "outside the brain": return group
    if group in SPECIAL_DIRECT: return place(group)
    lobe = PARENT_LOBE.get(group, group) if extra else group   # extra-axial is named after the lobe, not its white matter
    loc = place(lobe)
    if extra: return f"extra-axial, near {'' if loc.startswith('both') else 'the '}{loc}"
    return loc


SPECIAL_DIRECT = {"sellar and suprasellar region", "cerebellopontine angle"}
PARENT_LOBE = {"frontal white matter": "frontal lobe", "parietal white matter": "parietal lobe", "temporal white matter": "temporal lobe",
               "occipital white matter": "occipital lobe", "insular white matter": "insula", "periventricular white matter": "lateral ventricle",
               "cerebral white matter": "cerebral hemisphere"}


def _resample_like_images(vol_1mm: np.ndarray, affine: np.ndarray, target_shape, target_spacing, posterior_shift_mm):
    """Nearest-neighbour copy of contrastive-pretraining/scripts/data.py load_and_resample_nii + crop_or_pad."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.Nifti1Image(vol_1mm.astype(np.int16), affine))
    data = np.asarray(img.dataobj).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    cur = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
    data = data.transpose(2, 0, 1)
    new = [int(data.shape[i] * cur[i] / target_spacing[i]) for i in range(3)]
    t = F.interpolate(torch.from_numpy(data)[None, None], size=new, mode="nearest")[0, 0].numpy()
    shift = int(round(posterior_shift_mm / target_spacing[2]))
    td, th, tw = target_shape; d, h, w = t.shape
    ds, hs = max((d - td) // 2, 0), max((h - th) // 2, 0)
    ws = min(max(w // 2 - shift - tw // 2, 0), max(w - tw, 0))
    t = t[ds:ds + td, hs:hs + th, ws:ws + tw]
    pads = [(td - t.shape[0]) // 2, (th - t.shape[1]) // 2, (tw - t.shape[2]) // 2]
    out = np.zeros(target_shape, np.int16)
    out[pads[0]:pads[0] + t.shape[0], pads[1]:pads[1] + t.shape[1], pads[2]:pads[2] + t.shape[2]] = t.astype(np.int16)
    return out


def build_region_map(target_shape=(256, 384, 384), target_spacing=(1.0, 0.5, 0.5), posterior_shift_mm=15.0,
                     patch=(2, 16, 16), atlas_path: str | Path | None = None, labels_path: str | Path | None = None,
                     brain_fraction=0.35, near_mm=20.0, periventricular_mm=6.0) -> RegionMap:
    import nibabel as nib
    atlas_path = Path(atlas_path or ASSETS / "nvseg_brain_mni_majority.nii.gz")
    labels = {int(k): v for k, v in json.loads(Path(labels_path or ASSETS / "nvseg_brain_labels.json").read_text()).items()}
    img = nib.load(str(atlas_path)); lab = np.asarray(img.dataobj).astype(np.int32)
    # group map in the 1 mm atlas grid: id = 1 + index into `keys`
    keys: list[tuple[str, str]] = []
    gid = np.zeros_like(lab)
    for code, name in labels.items():
        key = _group_of(name)
        if key not in keys: keys.append(key)
        gid[lab == code] = keys.index(key) + 1
    # white matter -> nearest lobe of the same side; periventricular band first
    wm = np.isin(gid, [keys.index(k) + 1 for k in keys if k[0] == "cerebral white matter"])
    vent = np.isin(gid, [keys.index(k) + 1 for k in keys if k[0] == "lateral ventricle"])
    lobe_ids = [keys.index(k) + 1 for k in keys if k[0] in LOBES]
    lobe = np.isin(gid, lobe_ids)
    _, (ia, ib, ic) = ndi.distance_transform_edt(~lobe, return_indices=True)
    near_lobe = gid[ia, ib, ic]
    dist_vent = ndi.distance_transform_edt(~vent)
    out = gid.copy()
    for side in ("left", "right"):
        pv = ("periventricular white matter", side)
        if pv not in keys: keys.append(pv)
        m = wm & (dist_vent <= periventricular_mm) & np.isin(gid, [keys.index(("cerebral white matter", side)) + 1])
        out[m] = keys.index(pv) + 1
    for lid in lobe_ids:
        lname, lside = keys[lid - 1]
        k = (lname.replace(" lobe", "").replace("insula", "insular") + " white matter", lside)
        if k not in keys: keys.append(k)
        m = wm & (out == gid) & (near_lobe == lid) & np.isin(gid, [keys.index(("cerebral white matter", lside)) + 1])
        out[m] = keys.index(k) + 1
    # special extra-axial regions, defined from neighbouring anatomy in the 1 mm atlas (not from any lesion data)
    brain = gid > 0
    ids = lambda name: [keys.index(k) + 1 for k in keys if k[0] == name]
    d_of = lambda name: ndi.distance_transform_edt(~np.isin(gid, ids(name)))
    ras = np.stack(np.meshgrid(*[np.arange(n) for n in gid.shape], indexing="ij"), -1) @ img.affine[:3, :3].T + img.affine[:3, 3]
    xs, ys, zs = ras[..., 0], ras[..., 1], ras[..., 2]
    mid = float(xs[brain].mean())
    vdc, bs = np.isin(gid, ids("hypothalamic and suprasellar region")), np.isin(gid, ids("brainstem"))
    bs_front = float(np.percentile(ys[bs], 95)); vdc_floor = float(np.percentile(zs[vdc], 5))
    sella = (~brain) & (np.abs(xs - mid) <= 12) & (zs <= vdc_floor + 3) & (zs >= vdc_floor - 25) & (ys >= bs_front - 2) & (ys <= bs_front + 25)
    bz_lo, bz_hi = float(np.percentile(zs[bs], 5)), float(np.percentile(zs[bs], 85))   # pontomedullary junction to upper pons
    cpa = (~brain) & (d_of("brainstem") <= 25) & (d_of("cerebellar hemisphere") <= 20) & (np.abs(xs - mid) >= 8) & (zs >= bz_lo) & (zs <= bz_hi)
    for key, m in ((("sellar and suprasellar region", "midline"), sella),
                   (("cerebellopontine angle", "left"), cpa & (xs < mid)), (("cerebellopontine angle", "right"), cpa & (xs >= mid))):
        keys.append(key); out[m] = len(keys)
    special_ids = {keys.index(k) + 1 for k in keys if k[0] in SPECIAL_DIRECT}
    # to the MIL image grid (nearest), then to tokens
    g = _resample_like_images(out, img.affine, target_shape, target_spacing, posterior_shift_mm)
    tgrid = tuple(s // p for s, p in zip(target_shape, patch))
    blocks = g.reshape(tgrid[0], patch[0], tgrid[1], patch[1], tgrid[2], patch[2]).transpose(0, 2, 4, 1, 3, 5).reshape(-1, int(np.prod(patch)))
    K = len(keys) + 1
    counts = np.zeros((blocks.shape[0], K), np.int32)
    for k in range(K): counts[:, k] = (blocks == k).sum(1)
    brain_cols = [k for k in range(1, K) if k not in special_ids]
    special_cols = sorted(special_ids)
    bfrac = counts[:, brain_cols].sum(1) / blocks.shape[1]
    intra = np.where(bfrac >= brain_fraction, np.asarray(brain_cols)[counts[:, brain_cols].argmax(1)] - 1, -1)
    sp = np.asarray(special_cols)[counts[:, special_cols].argmax(1)] - 1
    sp = np.where(counts[:, special_cols].max(1) >= 0.25 * blocks.shape[1], sp, -1)
    tok_mm = np.array([target_spacing[i] * patch[i] for i in range(3)])
    inside = (intra >= 0).reshape(tgrid)
    dist, (ja, jb, jc) = ndi.distance_transform_edt(~inside, sampling=tok_mm, return_indices=True)
    nearest = intra.reshape(tgrid)[ja, jb, jc].reshape(-1)
    n = len(keys)
    all_keys = [(k[0], k[1], False) for k in keys] + [(k[0], k[1], True) for k in keys] + [("outside the brain", "none", True)]
    tr = np.where(intra >= 0, intra,
         np.where(sp >= 0, sp,
         np.where(dist.reshape(-1) <= near_mm, n + nearest, 2 * n)))
    return RegionMap(token_region=tr.astype(np.int16), keys=all_keys, grid=tgrid)


def region_masses(region_map: RegionMap, idx: np.ndarray, positive_contrib: np.ndarray) -> np.ndarray:
    """(n_classes, n_regions) positive evidence mass per region. idx: flat global-grid token index per token row."""
    rid = region_map.token_region[idx].astype(np.int64)
    out = np.zeros((positive_contrib.shape[1], len(region_map.names)), np.float64)
    for c in range(positive_contrib.shape[1]):
        out[c] = np.bincount(rid, weights=positive_contrib[:, c].astype(np.float64), minlength=len(region_map.names))
    return out


def describe(masses: np.ndarray, keys: list[tuple[str, str, bool]], cover=0.6, max_regions=2, multifocal_top=0.2,
             bilateral_min=0.3) -> str:
    """Plain-language location from one class's region masses (shares of positive evidence)."""
    total = float(masses.sum())
    if total <= 0: return ""
    share = masses / total
    l = sum(float(share[i]) for i, k in enumerate(keys) if k[1] == "left")
    r = sum(float(share[i]) for i, k in enumerate(keys) if k[1] == "right")
    bilateral = (l + r) >= 0.5 and min(l, r) >= bilateral_min * (l + r)
    # aggregate over side when bilateral, otherwise keep the side
    # merge intra- and extra-axial parts of a region; call it extra-axial only when most of its evidence is outside the brain
    tot: dict[tuple, float] = {}; ext: dict[tuple, float] = {}
    for i, (g, s_, e) in enumerate(keys):
        g2 = PARENT_LOBE.get(g, g) if e else g
        key = (g2, "both" if bilateral and s_ in ("left", "right") else s_)
        tot[key] = tot.get(key, 0.0) + float(share[i]); ext[key] = ext.get(key, 0.0) + (float(share[i]) if e else 0.0)
    agg = {(g, s_, ext[(g, s_)] > 0.5 * tot[(g, s_)]): v for (g, s_), v in tot.items()}
    order = sorted(agg, key=lambda k: -agg[k])
    if agg[order[0]] < multifocal_top:
        side = "bilateral" if bilateral else ("left-sided" if l > r else "right-sided") if l + r >= 0.5 else "midline"
        return f"multifocal, {side}, most in the {_text(order[0])}"
    picked, acc = [], 0.0
    for k in order[:max_regions]:
        picked.append(_text(k)); acc += agg[k]
        if acc >= cover: break
    return " and ".join(picked)


def _text(k) -> str:
    g, s_, e = k
    return phrase(g, s_, e, bilateral=(s_ == "both"))
