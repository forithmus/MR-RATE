"""Whole-volume three-component PCA previews, with exact tiled token geometry."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .data import raw_volume_with_support, is_mri_sequence
from .release import load_release


@torch.inference_mode()
def encode_volume(model, volume, tile, output, patch=(2, 16, 16), support=None, cache_only=False):
    if support is None or tuple(support.shape) != tuple(volume.shape):
        raise ValueError("Original MRI support mask is required")
    background = float(volume.float().min())
    shape = tuple(volume.shape)
    pads = []
    for n, t in reversed(list(zip(shape, tile))):
        pads.extend((0, (-n) % t))
    padded = F.pad(volume.float(), pads, value=0)
    valid = F.pad(support.float(), pads, value=0)
    grid = tuple(n // p for n, p in zip(padded.shape, patch))
    features = None if cache_only else np.zeros((*grid, model.embed_dim), dtype=np.float16)
    foreground = np.zeros(grid, dtype=bool)
    chunks = []
    latent_path = output.with_suffix(".latents.f16")
    tile_grid = tuple(t // p for t, p in zip(tile, patch))
    offset = 0
    with latent_path.open("wb") as sink:
        for z in range(0, padded.shape[0], tile[0]):
            for y in range(0, padded.shape[1], tile[1]):
                for x in range(0, padded.shape[2], tile[2]):
                    crop = padded[z:z+tile[0], y:y+tile[1], x:x+tile[2]][None, None].cuda().bfloat16()
                    supported = valid[z:z+tile[0], y:y+tile[1], x:x+tile[2]][None,None].cuda()
                    fg = F.max_pool3d(supported, patch, patch)[0,0] > 0
                    if not bool(fg.any()):
                        continue
                    tokens, g = model.prepare_tokens_with_masks(crop)
                    rope = tuple(c.to(device=tokens.device, dtype=tokens.dtype)
                                 for c in model.rope_embed(D=g[0], H=g[1], W=g[2]))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for block in model.blocks[:-1]:
                            tokens = block(tokens, rope)
                        latent = tokens.to(dtype=torch.float16).cpu().numpy()[0]
                        sink.write(latent.tobytes())
                        if not cache_only:
                            tokens = model.blocks[-1](tokens, rope)
                            patch_tokens = model.norm(tokens)[:, 1+model.n_storage_tokens:]
                    start = tuple(v//p for v, p in zip((z,y,x), patch))
                    region = tuple(slice(a,a+n) for a,n in zip(start,tile_grid))
                    if not cache_only:features[region] = patch_tokens.reshape(*tile_grid, model.embed_dim).float().cpu().numpy()
                    foreground[region] = fg.cpu().numpy()
                    chunks.append({"offset": offset, "length": len(latent), "grid": list(g),
                                   "start": list(start)})
                    offset += len(latent)
    output.with_suffix(".cache.json").write_text(json.dumps({
        "format": "mrdino_penultimate_tiles_v1", "dim": model.embed_dim,
        "prefix": 1+model.n_storage_tokens, "latent_file": latent_path.name,
        "foreground_file": output.with_suffix(".foreground.npy").name,
        "foreground_tokens": int(foreground.sum()),
        "support_rule": "nonzero original resampled MRI before zscore; original training image unchanged",
        "chunks": chunks, "token_grid": grid, "volume_shape": shape}))
    if not cache_only:np.save(output.with_suffix(".tokens.npy"), features)
    np.save(output.with_suffix(".foreground.npy"), foreground)
    return features, foreground


def render(volume, features, foreground, out, title):
    values = features[foreground].astype(np.float32)
    if len(values) < 3:
        raise ValueError("Fewer than three foreground tokens")
    pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
    scores = pca.fit_transform(values)
    lo, hi = np.percentile(scores, [1,99], axis=0)
    rgb = np.zeros((*foreground.shape,3), np.float32)
    rgb[foreground] = np.clip((scores-lo)/(hi-lo+1e-8),0,1)
    np.savez(out.with_suffix(".pca.npz"), mean=pca.mean_, components=pca.components_,
             explained_variance_ratio=pca.explained_variance_ratio_, rgb=rgb,
             lower=lo, upper=hi)
    image = volume.float().numpy()
    spacing=np.array([1.,.5,.5]);patch=np.array([2,16,16]);pitch=spacing*patch
    center = np.median(np.argwhere(foreground),axis=0).astype(int)
    fig, axes = plt.subplots(2,3,figsize=(13,8),facecolor="#111827")
    for axis in range(3):
        sl = [slice(None)]*3; sl[axis] = int(center[axis]); sl=tuple(sl)
        image_sl=[slice(None)]*3
        image_sl[axis]=min(int(center[axis]*patch[axis]+patch[axis]//2),image.shape[axis]-1)
        remaining=[i for i in range(3) if i!=axis]
        axes[0,axis].imshow(image[tuple(image_sl)].T,cmap="gray",origin="lower",aspect=spacing[remaining[1]]/spacing[remaining[0]])
        axes[1,axis].imshow(rgb[sl].transpose(1,0,2),origin="lower",aspect=pitch[remaining[1]]/pitch[remaining[0]])
        axes[0,axis].set_title(f"Input: axis {axis}, index {center[axis]}",color="white")
        axes[1,axis].set_title("Whole-volume PCA → RGB",color="white")
        for row in range(2): axes[row,axis].axis("off")
    fig.suptitle(title,color="white");fig.tight_layout();fig.savefig(out.with_suffix(".png"),dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--release",required=True);p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(0)
    model,meta=load_release(args.release,"cuda")
    train=meta["args"];tile=tuple(meta["config"]["training_crops"]["global"])
    panels=[]
    for row in json.loads(Path(args.manifest).read_text()):
        files=[f for f in row["nifti_files"] if is_mri_sequence(f)]
        uid=row["study_uid"];sequences=[]
        for number,source in enumerate(files):
            stem=f"{uid}_s{number:03d}"
            print(uid,"preprocessing",source,flush=True)
            volume,support=raw_volume_with_support(source,tuple(train["target_shape"]),tuple(train["target_spacing"]),train["posterior_shift_mm"])
            features,fg=encode_volume(model,volume,tile,out/stem,support=support)
            render(volume,features,fg,out/stem,uid+" · "+Path(source).name)
            sequences.append({"cache":stem+".cache.json","source":source})
            panels.append(f'<article><h2>{html.escape(uid)}</h2><p>{html.escape(Path(source).name)}</p><img src="{stem}.png"></article>')
            print(uid,"PCA saved",int(fg.sum()),"foreground tokens",flush=True)
            (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>MR DINO embeddings</title><style>body{background:#111827;color:#eee;font:16px sans-serif;max-width:1200px;margin:auto}img{width:100%}</style><h1>MR DINO: whole-volume PCA</h1><p>Three-component PCA fitted independently to all foreground tokens of each volume. Non-overlapping tiles assembled before PCA. Colors are independent between sequences. Prototype heads are absent from the inference releases.</p>'+''.join(panels))
        (out/f"{uid}.study.json").write_text(json.dumps({"study_uid":uid,"release_config":meta["config"],"sequences":sequences},indent=2))
    (out/"provenance.json").write_text(json.dumps(meta,indent=2))
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>MR DINO embeddings</title><style>body{background:#111827;color:#eee;font:16px sans-serif;max-width:1200px;margin:auto}img{width:100%}</style><h1>MR DINO: whole-volume PCA</h1><p>Three-component PCA fitted independently to all foreground tokens of each volume. Non-overlapping inference tiles are assembled before PCA. Colors are independent between scans. Learned prototype heads are absent from these inference releases.</p>'+''.join(panels))


if __name__=="__main__":main()
