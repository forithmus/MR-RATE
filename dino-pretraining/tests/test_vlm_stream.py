"""Compare the MRI port of MR-RATE with dense autograd, including two ranks."""
import copy
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from mr_dino.vlm_core import StreamPool,pooled_logits_reference,contrastive_ce,streamed_step


def test_stream_pool_matches_dense_values_and_every_gradient():
    torch.manual_seed(8)
    visual=torch.nn.Linear(7,5);text=torch.nn.Linear(7,5)
    visual_ref=copy.deepcopy(visual);text_ref=copy.deepcopy(text)
    log_t=torch.tensor(1.2,requires_grad=True);log_p=torch.tensor(.8,requires_grad=True)
    ref_t=log_t.detach().clone().requires_grad_();ref_p=log_p.detach().clone().requires_grad_()
    image=torch.randn(17,7);words=torch.randn(4,7)
    t=F.normalize(text(words),dim=-1);pt=log_p.exp()
    pool=StreamPool(t,pt.detach())
    with torch.no_grad():
        for x in image.split(6):pool.add(visual(x))
    p=pool.pooled().requires_grad_(True)
    logits=(F.normalize(p,dim=-1)*t).sum(1)*log_t.exp()
    loss=logits.square().mean();loss.backward(retain_graph=True)
    t_leaf=t.detach().requires_grad_();p_leaf=pt.detach().requires_grad_()
    for x in image.split(6):pool.surrogate(t_leaf,p_leaf,visual(x),p.detach(),p.grad).backward()
    torch.autograd.backward((t,pt),(t_leaf.grad,p_leaf.grad))
    dense=pooled_logits_reference(F.normalize(text_ref(words),dim=-1),visual_ref(image),ref_p.exp(),ref_t.exp())
    dense.square().mean().backward()
    torch.testing.assert_close(logits,dense)
    for a,b in zip(list(visual.parameters())+list(text.parameters())+[log_t,log_p],
                   list(visual_ref.parameters())+list(text_ref.parameters())+[ref_t,ref_p]):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=3e-5)


def networks():
    torch.manual_seed(17)
    return torch.nn.Linear(6,4),torch.nn.Linear(6,4),torch.nn.Parameter(torch.tensor(1.2)),torch.nn.Parameter(torch.tensor(.8))


def data(rank):
    generator=torch.Generator().manual_seed(300+rank)
    return torch.randn(3+4*rank,6,generator=generator),torch.randn(2,6,generator=generator)


def distributed_worker(rank,init_file,result_file):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        visual,text,t,p=networks();image,words=data(rank)
        def factory():
            for chunk in image.split(3):yield visual(chunk)
        loss,_,_,_=streamed_step(F.normalize(text(words),dim=-1),torch.ones(2,dtype=torch.bool),factory,t,p,2)
        params=list(visual.parameters())+list(text.parameters())+[t,p]
        flat=torch.cat([x.grad.flatten() for x in params]);dist.all_reduce(flat);flat/=2
        if rank==0:torch.save({'loss':loss,'gradient':flat},result_file)
    finally:dist.destroy_process_group()


def test_two_rank_full_step_matches_dense_global_objective(tmp_path):
    result=tmp_path/'result.pt'
    mp.spawn(distributed_worker,args=(str(tmp_path/'rendezvous'),str(result)),nprocs=2,join=True)
    actual=torch.load(result,weights_only=True)
    visual,text,t,p=networks()
    rows=[data(rank) for rank in range(2)]
    sentences=F.normalize(text(torch.cat([r[1] for r in rows])),dim=-1)
    logits=torch.stack([pooled_logits_reference(sentences,visual(r[0]),p.exp(),t.exp()) for r in rows],dim=1)
    loss=contrastive_ce(logits,2,torch.ones(4,dtype=torch.bool),torch.device('cpu'))
    loss.backward()
    expected=torch.cat([x.grad.flatten() for x in list(visual.parameters())+list(text.parameters())+[t,p]])
    torch.testing.assert_close(actual['loss'],loss)
    torch.testing.assert_close(actual['gradient'],expected,atol=2e-5,rtol=3e-5)
