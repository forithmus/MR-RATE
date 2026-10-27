"""the two-pass VL-CABS training, ported to multi-sequence MR DINO.

Frozen prefix caches are shared with MRI MIL. The final block starts from DINO,
not from a MIL checkpoint. Online mode recomputes the same frozen prefix.
"""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, json, math, os, random, signal
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from .data import discover_raw_aligned
from .geometry import position_features
from .mil import study_factory
from .mil_stream import LastBlockAdapter
from .release import load_release
from .vision import online_study_factory
from .vlm_core import TextEncoder,VisualProjector,streamed_step


def setup():
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    cpu=os.environ.get('MRRATE_VLM_DEVICE')=='cpu'
    device=torch.device('cpu') if cpu else torch.device('cuda',local)
    if not cpu:torch.cuda.set_device(local)
    if world>1:dist.init_process_group('gloo' if cpu else 'nccl',device_id=None if cpu else device,timeout=timedelta(minutes=30))
    return rank,world,local,device


def barrier(device):
    if dist.is_initialized():
        dist.barrier() if device.type=='cpu' else dist.barrier(device_ids=[device.index])


def read_sentences(path):
    result={}
    with open(path) as stream:
        for line_number,line in enumerate(stream,1):
            if line.startswith('version https://git-lfs'):
                raise ValueError('Sentences are a Git LFS pointer; run git lfs pull for this file')
            row=json.loads(line)
            if not row.get('valid_json',False):continue
            sentences=row.get('extracted_sentences',[])
            if not sentences:continue
            uid=str(row['volume_name'])
            if uid in result:raise ValueError(f'Duplicate sentence study at line {line_number}: {uid}')
            if not all(isinstance(s,str) and s.strip() for s in sentences):raise ValueError(f'Invalid sentences for {uid}')
            result[uid]=sentences
    if not result:raise ValueError('No usable finding sentences')
    return result


def backend(config,device):
    online=config.get('mode','offline')=='online'
    backbone,metadata=load_release(config['release'],device if online else 'cpu')
    adapter=LastBlockAdapter(backbone).to(device)
    with open(config['splits_csv'],newline='') as stream:rows=list(csv.DictReader(stream))
    membership={r['study_uid']:r['split'] for r in rows}
    patients={s:{r['patient_uid'] for r in rows if r['split']==s} for s in ('train','val','test')}
    if any(patients[a]&patients[b] for a,b in (('train','val'),('train','test'),('val','test'))):raise ValueError('Patient split leakage')
    if online:
        studies={s['study_uid']:s for s in discover_raw_aligned(config['data_folder'],set(membership),metadata['args']['space'])}
    else:
        del backbone
        studies={}
        for path in sorted(Path(config['cache_dir']).glob('*.study.json')):
            row=json.loads(path.read_text());uid=row['study_uid']
            if row['release_config']!=metadata['config']:raise ValueError('Cache belongs to another DINO checkpoint')
            if uid in studies:raise ValueError('Duplicate MRI cache')
            studies[uid]=path
    def factory(uid):
        if online:return online_study_factory(studies[uid],backbone,adapter,metadata,device,return_geometry=True)
        return study_factory(studies[uid],adapter,None,device,return_geometry=True)
    if not studies:raise ValueError('No MR studies available')
    if dist.is_initialized():
        signature=hashlib.sha256(json.dumps({'ids':sorted(studies),'release':metadata['config']},sort_keys=True).encode()).hexdigest()
        signatures=[None]*dist.get_world_size();dist.all_gather_object(signatures,signature)
        if len(set(signatures))!=1:raise ValueError('Ranks see different MRI cohorts/releases; use a shared complete cache or image tree')
    return adapter,metadata,membership,studies,factory


def normalize(config,output,rank,world,device):
    adapter,metadata,membership,studies,factory=backend(config,device)
    ids=sorted(u for u in studies if membership.get(u)=='train')
    missing=[u for u in membership if membership[u]=='train' and u not in studies]
    if missing and not config.get('allow_incomplete_data',False):raise ValueError(f'Incomplete training normalization cohort: {len(missing)} missing MRI studies')
    sums=torch.zeros(adapter.dim,device=device,dtype=torch.float64)
    squared=torch.zeros((),device=device,dtype=torch.float64);count=torch.zeros_like(squared)
    with torch.no_grad():
        for uid in ids[rank::world]:
            for y,_,_,_ in factory(uid)():
                sums+=y.double().sum(0);squared+=y.double().square().sum();count+=len(y)
    if world>1:
        for value in (sums,squared,count):dist.all_reduce(value)
    if count<=0:raise ValueError('Empty training normalization population')
    mean=sums/count
    rms=((squared/count-mean.square().sum())/adapter.dim).clamp_min(1e-12).sqrt()
    if rank==0:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        state={'mean':mean.float().cpu(),'rms':float(rms),'release_config':metadata['config'],
               'split':'train','studies':len(ids),'tokens':int(count)}
        pending=path.with_suffix('.pending');torch.save(state,pending);pending.replace(path)
        print({'event':'normalization_complete','studies':len(ids),'tokens':int(count),'rms':float(rms)},flush=True)
    barrier(device)


def batches(ids,world,seed=None):
    order=list(ids)
    if seed is not None:random.Random(seed).shuffle(order)
    for start in range(0,len(order),world):yield order[start:start+world]+[None]*max(0,start+world-len(order))


def train(config,output,rank,world,local,device,max_steps=0):
    if world<2:raise ValueError('MR-RATE VLM uses one study per rank and needs at least two distributed ranks for negatives')
    cfg=config['training'];spec=cfg['model'];seed=cfg['seed'];torch.manual_seed(seed)
    adapter,metadata,membership,studies,raw_factory=backend(config,device)
    sentences=read_sentences(config['sentences'])
    train_ids=sorted(u for u in studies if membership.get(u)=='train' and u in sentences)
    val_ids=sorted(u for u in studies if membership.get(u)=='val' and u in sentences)
    if not train_ids or not val_ids:raise ValueError('Training and validation MRI/sentence cohorts are required')
    missing=[u for u in sentences if membership.get(u) in ('train','val') and u not in studies]
    if missing and not config.get('allow_incomplete_data',False):raise ValueError(f'Incomplete VLM cohort: {len(missing)} train/val MRI studies missing')
    norm=torch.load(config['normalization'],map_location=device,weights_only=False)
    if norm['release_config']!=metadata['config'] or norm['split']!='train':raise ValueError('Wrong normalization provenance')
    mean=norm['mean'].float();rms=float(norm['rms'])
    if mean.shape!=(adapter.dim,) or not math.isfinite(rms) or rms<=0:raise ValueError('Invalid MRI normalization')
    frequencies=cfg.get('position_freqs',4);dim=spec['dim']
    visual=VisualProjector(adapter.dim+6*frequencies,dim).to(device)
    text=TextEncoder(spec['text_model'],dim).to(device)
    from transformers import BertTokenizer
    tokenizer=BertTokenizer.from_pretrained(spec['text_model'],do_lower_case=True)
    temperature=torch.nn.Parameter(torch.tensor(math.log(1/.07),device=device))
    pool_temperature=torch.nn.Parameter(torch.tensor(math.log(spec.get('pool_temperature_init',1/.07)),device=device))
    groups=[{'params':[p for p in adapter.parameters() if p.requires_grad and p.ndim>1],'lr':spec['lr_block'],'weight_decay':spec['wd']},
            {'params':[p for p in adapter.parameters() if p.requires_grad and p.ndim<=1],'lr':spec['lr_block'],'weight_decay':0.},
            {'params':visual.parameters(),'lr':spec['lr_proj'],'weight_decay':spec['wd']},
            {'params':text.bert.parameters(),'lr':spec['lr_text'],'weight_decay':spec['wd']},
            {'params':text.proj.parameters(),'lr':spec['lr_proj'],'weight_decay':spec['wd']},
            {'params':[temperature,pool_temperature],'lr':spec['lr_temp'],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups)
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    base_lrs=[g['lr'] for g in optimizer.param_groups]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    epoch0=batch0=step=0;best=float('inf')
    latest=out/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['config']!=config or state['world']!=world or state['release_config']!=metadata['config']:raise ValueError('Resume recipe/world/release mismatch')
        adapter.load_state_dict(state['adapter']);visual.load_state_dict(state['visual']);text.load_state_dict(state['text']);optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():temperature.copy_(state['temperature']);pool_temperature.copy_(state['pool_temperature'])
        epoch0,batch0,step,best=state['epoch'],state['batch'],state['step'],state['best_val_loss']
        rng=state['rng'][rank];torch.set_rng_state(rng['torch'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'].cpu(),device)
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    else:
        torch.manual_seed(seed+rank);random.seed(seed+rank);np.random.seed(seed+rank)
    stopping=False;requested=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    def request(*_):
        nonlocal requested
        requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,request)
    def save(epoch,batch,name='latest.pt'):
        rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
             'python':random.getstate(),'numpy':np.random.get_state()}
        all_rng=[None]*world;dist.all_gather_object(all_rng,rng)
        if rank==0:
            state={'format':'mrdino_vlm_v1','config':config,'world':world,'release_config':metadata['config'],
                   'epoch':epoch,'batch':batch,'step':step,'best_val_loss':best,'adapter':adapter.state_dict(),
                   'visual':visual.state_dict(),'text':text.state_dict(),'temperature':temperature.detach(),
                   'pool_temperature':pool_temperature.detach(),'optimizer':optimizer.state_dict(),'rng':all_rng}
            pending=out/(name+'.pending');torch.save(state,pending);pending.replace(out/name)
        barrier(device)
    def visual_factory(uid):
        if uid is None:
            def empty():yield visual(torch.zeros(1,adapter.dim+6*frequencies,device=device),0)
            return empty
        raw=raw_factory(uid)
        def chunks():
            for y,_,coordinates,kind in raw():
                features=torch.cat(((y-mean)/rms,position_features(coordinates,frequencies)*spec.get('position_scale',1.)),dim=1)
                yield visual(features,kind).float()
        return chunks
    slots=cfg['max_sentences'];total=math.ceil(len(train_ids)/world)*cfg['epochs']
    def run_case(uid,training):
        values=list(sentences[uid]) if uid else []
        chooser=random.Random(seed*7919+step*131+rank)
        if len(values)>slots:values=chooser.sample(values,slots)
        n=len(values);values+=['']*(slots-n)
        mask=torch.arange(slots,device=device)<n
        encoded=tokenizer(values,padding=True,truncation=True,max_length=cfg.get('max_tokens',128),return_tensors='pt').to(device)
        vectors=text(encoded.input_ids,encoded.attention_mask)
        vectors=torch.where(mask[:,None],vectors,torch.zeros_like(vectors))
        return streamed_step(vectors,mask,visual_factory(uid),temperature,pool_temperature,slots,real=uid is not None,backward=training)
    if rank==0:
        (out/'coverage.json').write_text(json.dumps({'train_studies':len(train_ids),'val_studies':len(val_ids),
            'test_studies_used':0,'all_sequences':True,'token_cap':None,'initial_final_block':'DINO release, not MIL'},indent=2))
    for epoch in range(epoch0,cfg['epochs']):
        adapter.train();visual.train();text.train()
        for batch,group in enumerate(batches(train_ids,world,seed+epoch)):
            if epoch==epoch0 and batch<batch0:continue
            fraction=min(1.,(step+1)/max(1,cfg['warmup_steps']))*(.5+.5*math.cos(math.pi*max(0,step-cfg['warmup_steps'])/max(1,total-cfg['warmup_steps'])))
            for g,base in zip(optimizer.param_groups,base_lrs):g['lr']=base*fraction
            optimizer.zero_grad(set_to_none=True)
            loss,logits,mask,count=run_case(group[rank],True)
            for p in parameters:
                if p.grad is None:p.grad=torch.zeros_like(p)
            flat=torch.cat([p.grad.reshape(-1) for p in parameters]);dist.all_reduce(flat);flat.div_(world)
            offset=0
            for p in parameters:p.grad.copy_(flat[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
            norm_value=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();step+=1
            stop_flag=torch.tensor(int(stopping or bool(max_steps and step>=max_steps)),device=device)
            request_flag=torch.tensor(int(requested),device=device)
            dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX);dist.all_reduce(request_flag,op=dist.ReduceOp.MAX)
            if rank==0:
                progress={'epoch':epoch+1,'batch':batch+1,'step':step,'loss':float(loss),'lr_fraction':fraction,
                          'gradient_norm':norm_value,'rank0_tokens':count,'temperature':float(temperature.detach().clamp(0,4).exp()),
                          'pool_temperature':float(pool_temperature.detach().clamp(0,4).exp())}
                (out/'progress.json').write_text(json.dumps(progress,indent=2));print(progress,flush=True)
            if step%cfg.get('save_every',200)==0 or stop_flag or request_flag:
                save(epoch,batch+1);requested=False
            if stop_flag:return
        adapter.eval();visual.eval();text.eval();numerator=denominator=correct=0.
        with torch.no_grad():
            for group in batches(val_ids,world):
                loss,logits,mask,_=run_case(group[rank],False)
                valid=int(mask.sum());numerator+=float(loss)*valid;denominator+=valid
                expected=torch.arange(world,device=device).repeat_interleave(slots)
                correct+=int(((logits.argmax(1)==expected)&mask).sum())
        val_loss=numerator/max(denominator,1)
        improved=val_loss<best;best=min(best,val_loss)
        if rank==0:
            result={'epoch':epoch+1,'step':step,'val_loss':val_loss,'sentence_to_report_accuracy':correct/max(denominator,1),'val_studies':len(val_ids)}
            (out/f'val_epoch_{epoch+1:03d}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
        save(epoch+1,0);save(epoch+1,0,f'epoch_{epoch+1:03d}.pt')
        if improved:save(epoch+1,0,'best.pt')
        batch0=0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['normalize','train'])
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--max-steps',type=int,default=0,help='Explicit smoke only; production default has no step cap')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    rank,world,local,device=setup()
    try:
        if args.task=='normalize':normalize(config,args.output,rank,world,device)
        else:train(config,args.output,rank,world,local,device,args.max_steps)
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
