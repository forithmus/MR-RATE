"""Cache all MRI sequences and train MR-RATE-style final-block + scalar MIL.

An inference release is initialization, never an optimizer resume. Route masks
are optional external MRI anatomy masks; no CT organ definitions are imported.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data import raw_volume_with_support, discover_raw_aligned, _load_split_ids, is_mri_sequence
from .mil_head import SparseClassifyThenAggregate
from .mil_stream import LastBlockAdapter, backward, predict
from .probe import auroc, read_labels
from .labels import DEFAULT_SCHEMA, read_final_labels
from .release import load_release, release_metadata


def cache(args):
    from .preview import encode_volume
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK",0)))
    model,meta=load_release(args.release,"cuda")
    cfg=meta["args"];out=Path(args.cache_dir);out.mkdir(parents=True,exist_ok=True)
    if args.input_manifest:
        rows=json.loads(Path(args.input_manifest).read_text())
    else:
        selected=_load_split_ids(args.splits_csv,args.split)
        studies=discover_raw_aligned(args.data_folder,selected,cfg["space"])
        rows=[{"study_uid":s["study_uid"],"nifti_files":s["image_paths"]} for s in studies]
    tile=tuple(meta["config"]["training_crops"]["global"])
    for row in rows[rank::world]:
        uid=row["study_uid"]
        if Path(uid).name!=uid:raise ValueError("Unsafe study UID")
        complete=out/f"{uid}.study.json"
        if complete.exists():
            previous=json.loads(complete.read_text())
            if previous["release_config"]!=meta["config"]:raise ValueError("Cache belongs to another release")
            continue
        sequences=[];token_count=0
        sources=[f for f in row["nifti_files"] if is_mri_sequence(f)]
        if not sources:raise ValueError(f"No MRI sequences for {uid}")
        for number,source in enumerate(sources):
            volume,support=raw_volume_with_support(source,tuple(cfg["target_shape"]),tuple(cfg["target_spacing"]),cfg["posterior_shift_mm"])
            stem=out/f"{uid}_s{number:03d}"
            _,foreground=encode_volume(model,volume,tile,stem,support=support,cache_only=True)
            token_count+=int(foreground.sum())
            sequences.append({"cache":stem.with_suffix(".cache.json").name,"source":source})
        pending=complete.with_suffix(".pending")
        pending.write_text(json.dumps({"study_uid":uid,"release_config":meta["config"],"sequences":sequences,"token_count":token_count},indent=2))
        pending.replace(complete)
        print(uid,"cached",len(sequences),"sequences",flush=True)


def study_factory(path,adapter,head,device,return_geometry=False):
    study=json.loads(path.read_text());sequences=[]
    for sequence in study["sequences"]:
        index_path=path.parent/sequence["cache"]
        index=json.loads(index_path.read_text())
        if index["dim"]!=adapter.dim or index["prefix"]!=adapter.prefix:
            raise ValueError("MRI token cache/model architecture mismatch")
        total=sum(c["length"] for c in index["chunks"])
        latent=index_path.parent/index["latent_file"]
        if latent.stat().st_size!=total*adapter.dim*2:raise ValueError("Incomplete latent cache")
        tokens=np.memmap(latent,mode="r",dtype=np.float16,shape=(total,adapter.dim))
        foreground=np.load(index_path.parent/index["foreground_file"],mmap_mode="r") if "foreground_file" in index else None
        sequences.append((index,tokens,foreground,sequence["source"]))
    def factory():
        for index,tokens,foreground,source in sequences:
            for chunk in index["chunks"]:
                start=chunk["offset"];length=chunk["length"]
                x=torch.from_numpy(np.array(tokens[start:start+length],copy=True)).to(device,dtype=torch.bfloat16 if device.type=="cuda" else torch.float32)
                if foreground is None:
                    fg=torch.tensor(chunk["foreground"],device=device,dtype=torch.bool)
                else:
                    region=tuple(slice(a,a+n) for a,n in zip(chunk["start"],chunk["grid"]))
                    fg=torch.tensor(np.asarray(foreground[region]).reshape(-1),device=device,dtype=torch.bool)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    y=adapter(x[None],chunk["grid"])[0,fg]
                if not len(y):continue
                route=None
                if "route_mask" in chunk and head is not None:
                    head.route_masks_seen=True
                    route=torch.tensor(chunk["route_mask"],device=device,dtype=torch.bool)[fg]
                    if route.shape!=(len(y),head.n_classes):raise ValueError("Wrong MRI route dimensions")
                if return_geometry:
                    from .geometry import normalized_positions,sequence_kind
                    coordinates=normalized_positions(chunk["start"],chunk["grid"],index["token_grid"],device)[fg]
                    yield y.float(),route,coordinates,sequence_kind(source)
                else:yield y.float(),route
    return factory


def train(args):
    local=int(os.environ.get("LOCAL_RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    if world>1:dist.init_process_group("nccl",device_id=device)
    rank=dist.get_rank() if world>1 else 0
    torch.manual_seed(args.seed)
    model,meta=load_release(args.release,device if args.mode=="online" else "cpu")
    adapter=LastBlockAdapter(model).to(device)
    if args.mode=="offline":del model
    names,labels=read_final_labels(args.labels_csv,args.label_schema)
    normalizer=torch.load(args.normalization,map_location=device,weights_only=False)
    if normalizer['release_config']!=meta['config'] or normalizer['split']!='train':raise ValueError('Wrong normalization provenance')
    mean=normalizer['mean'].float();rms=float(normalizer['rms'])
    if mean.shape!=(adapter.dim,) or not math.isfinite(rms) or rms<=0:raise ValueError('Invalid MRI normalization')
    head=SparseClassifyThenAggregate(adapter.dim+6*args.position_freqs,len(names),use_norm=False,checkpoint_chunks=False).to(device)
    # Soft MRI routes, when provided. Missing routes mean all tokens are eligible.
    head.route_bias_raw=torch.nn.Parameter(torch.full((len(names),),2.,device=device))
    params=list(adapter.parameters())+list(head.parameters())
    optimizer=torch.optim.AdamW([{"params":adapter.parameters(),"lr":args.block_lr},
                                 {"params":[p for n,p in head.named_parameters() if n!="route_bias_raw"],"lr":args.head_lr},
                                 {"params":[head.route_bias_raw],"lr":args.head_lr,"weight_decay":0.}],weight_decay=args.weight_decay)
    with open(args.splits_csv,newline="") as handle:rows=list(csv.DictReader(handle))
    splits={r["study_uid"]:r["split"] for r in rows}
    patient={r["study_uid"]:r["patient_uid"] for r in rows}
    patient_sets={s:{patient[u] for u in splits if splits[u]==s} for s in ("train","val","test")}
    if any(patient_sets[a]&patient_sets[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Patient leakage between MR-RATE splits")
    caches={}
    if args.mode=="offline":
        if not args.cache_dir:raise ValueError("Offline mode requires --cache-dir")
        for path in sorted(Path(args.cache_dir).glob("*.study.json")):
            row=json.loads(path.read_text());uid=row["study_uid"]
            if row["release_config"]!=meta["config"]:raise ValueError("Mixed checkpoints in token cache")
            if uid in caches:raise ValueError("Duplicate study cache")
            caches[uid]=path
    else:
        if not args.data_folder:raise ValueError("Online mode requires --data-folder")
        for study in discover_raw_aligned(args.data_folder,set(splits),meta["args"]["space"]):
            if study["study_uid"] in caches:raise ValueError("Duplicate online study")
            caches[study["study_uid"]]=study
    def make_factory(uid):
        if args.mode=="offline":raw=study_factory(caches[uid],adapter,head,device,return_geometry=True)
        else:
            from .vision import online_study_factory
            raw=online_study_factory(caches[uid],model,adapter,meta,device,return_geometry=True)
        def transformed():
            from .geometry import position_features
            for y,mask,coordinates,_ in raw():
                yield torch.cat(((y-mean)/rms,position_features(coordinates,args.position_freqs)*args.position_scale),1),mask
        return transformed
    train_ids=sorted(u for u in caches if splits.get(u)=="train" and u in labels)
    val_ids=sorted(u for u in caches if splits.get(u)=="val" and u in labels)
    if not train_ids or not val_ids:raise ValueError("Both train and validation caches are required")
    missing=[u for u in labels if splits.get(u) in ('train','val') and u not in caches]
    if missing and not args.allow_incomplete_data:raise ValueError(f'Incomplete MRI cohort: {len(missing)} labelled train/val studies missing; first={missing[:10]}')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=0
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if state["release_config"]!=meta["config"] or state["classes"]!=names:raise ValueError("Resume checkpoint mismatch")
        adapter.load_state_dict(state["adapter"]);head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"]);start=state["epoch"]
    def objective(logits,target):
        known=torch.isfinite(target)&(target>=0)
        if not bool(known.any()):raise ValueError("Study has no known labels")
        if not bool(((target[known]==0)|(target[known]==1)).all()):raise ValueError("Nonbinary pathology labels")
        return torch.nn.functional.binary_cross_entropy_with_logits(logits[known],target[known])
    for epoch in range(start,args.epochs):
        order=torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        steps=math.ceil(len(order)/world)
        order=order+[-1]*(steps*world-len(order))
        head.train();adapter.train();losses=[]
        for index in order[rank::world]:
            optimizer.zero_grad(set_to_none=True)
            real=index>=0
            if real:
                uid=train_ids[index]
                target=torch.tensor(labels[uid],device=device)
                loss,_=backward(head,make_factory(uid),target,objective)
                losses.append(float(loss))
            if world>1:
                count=torch.tensor(float(real),device=device)
                dist.all_reduce(count)
                for param in params:
                    if param.grad is None:param.grad=torch.zeros_like(param)
                    dist.all_reduce(param.grad);param.grad.div_(count.clamp_min(1))
            torch.nn.utils.clip_grad_norm_(params,3.,error_if_nonfinite=True)
            optimizer.step()
        head.eval();adapter.eval();predictions=[]
        for uid in val_ids[rank::world]:
            probability=predict(head,make_factory(uid)).sigmoid()[0].cpu().tolist()
            predictions.append({"study_uid":uid,"probabilities":probability})
        if world>1:
            gathered=[None]*world;dist.all_gather_object(gathered,predictions)
            predictions=[r for shard in gathered for r in shard]
        meter=torch.tensor([sum(losses),len(losses)],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(meter)
        routed=torch.tensor(int(getattr(head,"route_masks_seen",False)),device=device)
        if world>1:dist.all_reduce(routed,op=dist.ReduceOp.MAX)
        if rank==0:
            per_class={}
            for c,name in enumerate(names):
                valid=[r for r in predictions if np.isfinite(labels[r["study_uid"]][c]) and labels[r["study_uid"]][c]>=0]
                per_class[name]=auroc(np.array([labels[r["study_uid"]][c] for r in valid]),np.array([r["probabilities"][c] for r in valid])) if valid else None
            values=[v for v in per_class.values() if v is not None]
            result={"epoch":epoch+1,"train_loss":float(meter[0]/meter[1].clamp_min(1)),"macro_auroc":float(np.mean(values)) if values else None,"per_class_auroc":per_class,"predictions":predictions}
            (out/f"epoch_{epoch+1:03d}.json").write_text(json.dumps(result,indent=2))
            state={"format":"mrdino_mil_v1","epoch":epoch+1,"release_config":meta["config"],"classes":names,"adapter":adapter.state_dict(),"head":head.state_dict(),"optimizer":optimizer.state_dict(),"arguments":vars(args),"normalization":normalizer,"architecture":{"dim":head.dim,"n_classes":len(names),"hidden_dim":512,"mlp_hidden_dim":384,"use_norm":False},"route_masks_used":bool(routed)}
            pending=out/"checkpoint.pending";torch.save(state,pending);pending.replace(out/f"epoch_{epoch+1:03d}.pt")
            print({k:v for k,v in result.items() if k not in {"predictions","per_class_auroc"}},flush=True)
        if world>1:dist.barrier(device_ids=[local])
    if world>1:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    c=sub.add_parser("cache");c.add_argument("--release",required=True);c.add_argument("--cache-dir",required=True)
    source=c.add_mutually_exclusive_group(required=True);source.add_argument("--input-manifest");source.add_argument("--data-folder")
    c.add_argument("--splits-csv");c.add_argument("--split",default="train",choices=["train","val","test"])
    t=sub.add_parser("train");t.add_argument("--release",required=True);t.add_argument("--cache-dir")
    t.add_argument("--mode",default="offline",choices=["offline","online"]);t.add_argument("--data-folder")
    t.add_argument('--normalization',required=True);t.add_argument('--position-freqs',type=int,default=4);t.add_argument('--position-scale',type=float,default=1.)
    t.add_argument('--allow-incomplete-data',action='store_true',help='Explicit implementation smoke only; production requires complete labelled train/val coverage')
    t.add_argument("--labels-csv",required=True);t.add_argument("--splits-csv",required=True);t.add_argument("--out",required=True)
    t.add_argument('--label-schema',default=str(DEFAULT_SCHEMA),help='Final ordered 74-diagnosis MRI schema')
    t.add_argument("--head-lr",type=float,required=True);t.add_argument("--block-lr",type=float,required=True)
    t.add_argument("--weight-decay",type=float,default=.01);t.add_argument("--epochs",type=int,default=10)
    t.add_argument("--seed",type=int,default=42);t.add_argument("--resume")
    args=parser.parse_args()
    if args.command=="cache":cache(args)
    else:train(args)


if __name__=="__main__":main()
