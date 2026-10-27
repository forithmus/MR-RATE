"""Exact streamed VL-CABS pooling and multi-positive contrastive learning."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .geometry import SEQUENCE_KINDS


def gather(x,dim=0):
    return torch.cat(GatherLayer.apply(x),dim) if dist.is_initialized() else x


def streamed_step(text, sentence_mask, factory, temperature, pool_temperature, slots,
                  real=True, backward=True):
    """Same two-pass chain rule and global loss as the trainer."""
    all_text=gather(text)
    if dist.is_initialized():
        masks=[torch.empty_like(sentence_mask) for _ in range(dist.get_world_size())]
        dist.all_gather(masks,sentence_mask)
        all_mask=torch.cat(masks)
        valid_reports=torch.stack([m.any() for m in masks])
    else:
        all_mask=sentence_mask;valid_reports=sentence_mask.any()[None]
    temp=temperature.clamp(0.,4.).exp()
    pt=pool_temperature.clamp(0.,4.).exp()
    pool=StreamPool(all_text,pt.detach())
    count=0
    with torch.no_grad():
        for z in factory():pool.add(z.float());count+=len(z)
    if count==0:raise ValueError("VLM study has no visual tokens")
    pooled=pool.pooled().detach().requires_grad_(backward)
    column=(temp*(F.normalize(pooled,dim=-1)*all_text).sum(1)).clamp(-100.,100.)[:,None]
    if not real:column=column*0.-1e4
    logits=gather(column,1).masked_fill(~valid_reports[None,:],-1e4)
    loss=contrastive_ce(logits,slots,all_mask,all_text.device)
    if not backward:return loss.detach(),logits.detach(),all_mask,count
    loss.backward(retain_graph=True)
    g=pooled.grad.detach();p=pooled.detach()
    t_leaf=all_text.detach().requires_grad_(True)
    pt_leaf=pt.detach().requires_grad_(True)
    seen=0
    for z in factory():
        pool.surrogate(t_leaf,pt_leaf,z.float(),p,g).backward();seen+=len(z)
    if seen!=count:raise RuntimeError("VLM replay changed token count")
    torch.autograd.backward((all_text,pt),(t_leaf.grad,pt_leaf.grad))
    return loss.detach(),logits.detach(),all_mask,count


class VisualProjector(torch.nn.Module):
    def __init__(self,dim_in,dim):
        super().__init__();self.proj=torch.nn.Linear(dim_in,dim);self.view=torch.nn.Embedding(len(SEQUENCE_KINDS),dim)
        torch.nn.init.zeros_(self.view.weight)
    def forward(self,x,kind):return self.proj(x)+self.view.weight[kind]

class TextEncoder(torch.nn.Module):
    def __init__(self,path,dim):
        super().__init__()
        from transformers import BertModel
        self.bert=BertModel.from_pretrained(path);self.proj=torch.nn.Linear(self.bert.config.hidden_size,dim)
    def forward(self,ids,att):
        return F.normalize(self.proj(self.bert(ids,attention_mask=att).last_hidden_state[:,0]).float(),dim=-1)

class StreamPool:
    """Exact VL-CABS attention pooling over a token stream without storing the tokens (memory independent of report size).
    Pass 1 (no grad): online softmax over chunks -> per-sentence max m, normaliser S and pooled vector p = sum_k a_k z_k with
    a_k = softmax_k(pool_temp * cos(t, z_k)). Pass 2 (grad): for each recomputed chunk the surrogate
        J = sum_ik a_ik.detach() * (g_i . z_k) + sum_ik s_ik * [a_ik (g_i . z_k - g_i . p_i)].detach()
    back-propagates exactly dL/dz_k = a_k g + (dL/ds_k) ds_k/dz_k and dL/dt, dL/dpool_temp of the attention path, given g = dL/dp."""
    def __init__(self,text,pool_temp):
        self.t=text.detach();n=len(self.t);D_=self.t.shape[1];dev=self.t.device
        dt=self.t.dtype;self.pt=float(pool_temp);self.m=torch.full((n,),-torch.inf,device=dev,dtype=dt);self.S=torch.zeros(n,device=dev,dtype=dt);self.P=torch.zeros(n,D_,device=dev,dtype=dt)
    @torch.no_grad()
    def add(self,z):
        s=self.pt*(self.t@z.T)*z.norm(dim=-1).clamp_min(1e-12).reciprocal()[None]
        mn=torch.maximum(self.m,s.amax(1));c=torch.exp(self.m-mn);e=torch.exp(s-mn[:,None])
        self.S=self.S*c+e.sum(1);self.P=self.P*c[:,None]+e@z;self.m=mn
    def pooled(self):return self.P/self.S[:,None]
    def surrogate(self,t_leaf,pool_temp,z,p,g):
        s=pool_temp*(t_leaf@z.T)*z.norm(dim=-1).clamp_min(1e-12).reciprocal()[None]
        with torch.no_grad():
            a=torch.exp(s-self.m[:,None])/self.S[:,None];gz=g@z.T;ds=a*(gz-(g*p).sum(1,keepdim=True))
        return (a*(g@z.T)).sum()+(s*ds).sum()

def pooled_logits_reference(text,z,pool_temp,temp):
    """Plain autograd version (whole z in memory): the definition the streaming path must reproduce (tests only)."""
    w=torch.softmax(pool_temp*((text@z.T)*z.norm(dim=-1).clamp_min(1e-12).reciprocal()[None]),dim=1);pooled=w@z
    return (F.normalize(pooled,dim=-1)*text).sum(1)*temp

class GatherLayer(torch.autograd.Function):
    """All-gather with a DDP-safe, deterministic backward.

    torch.distributed.nn.functional.all_gather issues its backward collective
    (reduce_scatter) through the autograd-distributed machinery, whose op
    ordering interleaves nondeterministically with DDP's gradient all_reduces
    on the shared communicator -> NCCL same-communicator order mismatch that
    deadlocks backward at scale (observed: 64 ranks hang right after step 0).

    This is the battle-tested OpenCLIP pattern: forward does a plain blocking
    all_gather; backward does a single blocking all_reduce(SUM) of the stacked
    grads and returns this rank's slice. The collective sits at one fixed
    autograd node with identical ordering on every rank.
    """
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        grad_stack = torch.stack(grads, dim=0)
        dist.all_reduce(grad_stack, op=dist.ReduceOp.SUM)
        return grad_stack[dist.get_rank()]

def contrastive_ce(logits, num_sentences_per_image, mask_all, device):
        """Symmetric (text<->image) InfoNCE used by both the global and local
        terms. Each sentence's positive is its own study; negatives are other
        studies (and, for image->text, other studies' sentences)."""
        logits = logits.float()
        B_global = logits.shape[1]
        N_T = logits.shape[0]
        sentence_to_image_map = torch.arange(B_global, device=device).repeat_interleave(num_sentences_per_image)
        image_membership = F.one_hot(sentence_to_image_map, num_classes=B_global).bool()

        if mask_all is not None:
            mask_float = mask_all.float()
            loss_t_all = F.cross_entropy(logits, sentence_to_image_map, reduction='none')
            loss_t_all = torch.where(mask_all, loss_t_all, torch.zeros_like(loss_t_all))
            loss_t = loss_t_all.sum() / mask_float.sum().clamp(min=1.0)

            pos_logits = logits[torch.arange(N_T, device=device), sentence_to_image_map]
            not_from_image = ~image_membership
            valid_sentences = mask_all.view(N_T, 1).expand(N_T, B_global)
            logits_masked_for_neg = logits.clone()
            logits_masked_for_neg[~(not_from_image & valid_sentences)] = -1e4
            logsumexp_neg = torch.logsumexp(logits_masked_for_neg, dim=0)[sentence_to_image_map]
            log_denom = torch.logsumexp(torch.stack([pos_logits, logsumexp_neg], dim=1), dim=1)
            loss_i_all = -pos_logits + log_denom
            loss_i_all = torch.where(mask_all, loss_i_all, torch.zeros_like(loss_i_all))
            loss_i = loss_i_all.sum() / mask_float.sum().clamp(min=1.0)
        else:
            loss_t = F.cross_entropy(logits, sentence_to_image_map)
            pos_logits = logits[torch.arange(N_T, device=device), sentence_to_image_map]
            logits_masked_for_neg = logits.clone()
            logits_masked_for_neg[~(~image_membership)] = -1e4
            logsumexp_neg = torch.logsumexp(logits_masked_for_neg, dim=0)[sentence_to_image_map]
            log_denom = torch.logsumexp(torch.stack([pos_logits, logsumexp_neg], dim=1), dim=1)
            loss_i = (-pos_logits + log_denom).mean()
        return loss_t + loss_i
