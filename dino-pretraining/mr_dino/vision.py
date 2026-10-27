"""Image-backed, replayable MRI features using the same geometry as cached MIL."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import raw_volume_with_support


def online_study_factory(study, backbone, adapter, metadata, device,return_geometry=False):
    args=metadata["args"]
    tile=tuple(metadata["config"]["training_crops"]["global"])
    patch=tuple(metadata["config"]["backbone"]["patch_size"])
    volumes=[raw_volume_with_support(source,tuple(args["target_shape"]),
              tuple(args["target_spacing"]),args["posterior_shift_mm"])
             for source in study["image_paths"]]
    backbone.eval().requires_grad_(False)
    def factory():
        for sequence,(volume,support) in enumerate(volumes):
            pads=[]
            for n,t in reversed(list(zip(volume.shape,tile))):pads.extend((0,(-n)%t))
            image=F.pad(volume,pads,value=0)
            valid=F.pad(support.float(),pads,value=0)
            for z in range(0,image.shape[0],tile[0]):
                for y in range(0,image.shape[1],tile[1]):
                    for x in range(0,image.shape[2],tile[2]):
                        sl=(slice(z,z+tile[0]),slice(y,y+tile[1]),slice(x,x+tile[2]))
                        keep=F.max_pool3d(valid[sl][None,None],patch,patch).flatten()>0
                        if not bool(keep.any()):continue
                        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
                            crop=image[sl][None,None].to(device)
                            tokens,grid=backbone.prepare_tokens_with_masks(crop)
                            rope=tuple(r.to(device=tokens.device,dtype=tokens.dtype)
                                       for r in backbone.rope_embed(D=grid[0],H=grid[1],W=grid[2]))
                            for block in backbone.blocks[:-1]:tokens=block(tokens,rope)
                        with torch.autocast(device.type,dtype=torch.bfloat16):
                            features=adapter(tokens,grid)[0,keep.to(device)]
                        if return_geometry:
                            from .geometry import normalized_positions,sequence_kind
                            start=tuple(a//p for a,p in zip((z,y,x),patch))
                            global_grid=tuple(n//p for n,p in zip(image.shape,patch))
                            coordinates=normalized_positions(start,grid,global_grid,device)[keep.to(device)]
                            yield features.float(),None,coordinates,sequence_kind(study["image_paths"][sequence])
                        else:yield features.float(),None
    return factory
