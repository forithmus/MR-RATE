"""MRI sequence kinds and the global-grid Fourier position features."""
import math
from pathlib import Path
import torch

SEQUENCE_KINDS=("t1","t1ce","t2","flair","swi","t2star","dwi","adc","pd","stir","tof","other")


def sequence_kind(path):
    name=Path(path).name.lower()
    for token,kind in (("flair","flair"),("swi","swi"),("adc","adc"),("dwi","dwi"),
                       ("diff","dwi"),("stir","stir"),("tof","tof"),("t2star","t2star"),("t2*","t2star")):
        if token in name:return SEQUENCE_KINDS.index(kind)
    if "t1" in name:
        return SEQUENCE_KINDS.index("t1ce" if any(s in name for s in ("-ce","_ce","post","gad","contrast")) else "t1")
    if "t2" in name:return SEQUENCE_KINDS.index("t2")
    if "pd" in name:return SEQUENCE_KINDS.index("pd")
    return SEQUENCE_KINDS.index("other")


def normalized_positions(start,grid,global_grid,device):
    axes=[torch.arange(n,device=device,dtype=torch.float32)+s+.5 for s,n in zip(start,grid)]
    coordinates=torch.stack(torch.meshgrid(*axes,indexing="ij"),dim=-1).reshape(-1,3)
    return coordinates/torch.tensor(global_grid,device=device,dtype=torch.float32)


def position_features(coordinates,freqs):
    angle=coordinates[:,:,None]*(math.pi*2.**torch.arange(freqs,device=coordinates.device))[None,None,:]
    return torch.cat((angle.sin(),angle.cos()),dim=2).reshape(len(coordinates),6*freqs)
