import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.utils.checkpoint import checkpoint
from torchvision import transforms as T, utils
import torchvision

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from pathlib import Path
import copy
import math
import random
import numpy as np
from functools import partial
import torch.distributed.nn.functional as dist_nn_fun

from transformers import BertTokenizer, BertModel

# =================================================================================
# Helpers
# =================================================================================

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim=-1, eps=1e-4)

def cast_tuple(t, l=1):
    return t if isinstance(t, (tuple, list)) else (t,) * l

def all_gather_batch(x):
    if not dist.is_initialized():
        return x
    gathered_list = dist_nn_fun.all_gather(x)
    return torch.cat(gathered_list, dim=0)

class RearrangeImage(nn.Module):
    def forward(self, x):
        if x.ndim == 3:
            return x
        return rearrange(x, 'b (h w z) c -> b c h w z', h=16, w=16)

# =================================================================================
# Memory-Efficient Pooling Modules
# =================================================================================

class SimpleAttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        b, r, t, d = x.shape
        x_flat = rearrange(x, 'b r t d -> (b t) r d')
        q = self.query.expand(b * t, -1, -1)
        sim = einsum('b i d, b j d -> b i j', q, x_flat) * self.scale

        if mask is not None:
            mask_exp = mask.unsqueeze(1).repeat(1, t, 1).view(b * t, 1, r)
            sim = sim.masked_fill(~mask_exp, -1e4)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, x_flat).squeeze(1)
        out = rearrange(out, '(b t) d -> b t d', b=b)
        return self.norm(out)

class CrossAttnPool(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, text_query, mask=None):
        b, r, t, d = x.shape
        q_input = text_query.unsqueeze(1).expand(-1, t, -1)
        x_flat = rearrange(x, 'b r t d -> (b t) r d')
        q_flat = rearrange(q_input, 'b t d -> (b t) d').unsqueeze(1)

        q = self.to_q(q_flat)
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b r (h d) -> b h r d', h=self.num_heads)
        v = rearrange(v, 'b r (h d) -> b h r d', h=self.num_heads)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask_exp = mask.unsqueeze(1).repeat(1, t, 1).view(b * t, 1, 1, r)
            sim = sim.masked_fill(~mask_exp, -1e4)

        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)').squeeze(1)
        out = rearrange(out, '(b t) d -> b t d', b=b)
        return self.norm(self.to_out(out))

class GatedAttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, text_query, mask=None):
        b, r, t, d = x.shape
        text_exp = text_query.unsqueeze(1).unsqueeze(2).expand(-1, r, t, -1)
        combined = torch.cat([text_exp, x], dim=-1)
        gate_scores = self.gate_net(combined).squeeze(-1)

        if mask is not None:
            gate_scores = gate_scores.masked_fill(~mask.unsqueeze(-1), -1e4)

        weights = F.softmax(gate_scores, dim=1)
        out = einsum('b r t, b r t d -> b t d', weights, x)
        return self.norm(out)

# =================================================================================
# MRRATE (Main Model)
# =================================================================================

class MRRATE(nn.Module):
    def __init__(
            self,
            *,
            image_encoder=None,
            text_encoder=None,
            dim_text=512,
            dim_image=1408,
            dim_latent=256,
            num_text_tokens=28897,
            text_seq_len=256,
            downsample_image_embeds=False,
            extra_latent_projection=False,
            fusion_mode="mid_cnn",
            pooling_strategy="simple_attn",
            use_gradient_checkpointing=False,
            **kwargs
    ):
        super().__init__()
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent
        self.text_seq_len = text_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.fusion_mode = fusion_mode
        self.pooling_strategy = pooling_strategy
        self.text_pad_id = 0

        # Encoders
        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            raise ValueError("Please pass the updated image_encoder instance.")

        # Latent Projections
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        if downsample_image_embeds:
            dim_conv = 512
            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv3d(dim_conv, dim_conv, 4, stride=2, padding=1, bias=False, groups=dim_conv),
                nn.Conv3d(dim_conv, dim_latent, 1),
                Rearrange('b c h w z -> b (h w z c)'),
                nn.Linear(dim_image, dim_latent, bias=False)
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)

        # Pooling Modules
        if pooling_strategy == "simple_attn":
            self.recon_pool = SimpleAttnPool(dim_latent)
        elif pooling_strategy == "cross_attn":
            self.recon_pool = CrossAttnPool(dim_latent, num_heads=4)
        elif pooling_strategy == "gated":
            self.recon_pool = GatedAttnPool(dim_latent)
        else:
            raise ValueError(f"Unknown pooling_strategy: {pooling_strategy}")

        self.logit_temperature = nn.Parameter(torch.tensor([np.log(1 / 0.07)]))
        self.extra_latent_projection = extra_latent_projection
        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)

    def run_checkpoint(self, fn, *args):
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    # --- KEY LOGIC: Compute logits with token mask ---
    def _compute_logit_matrix(self, visual_tokens_all, text_latents_all, token_mask_all, temp, chunk_size=4):
        """
        Compute VL-CABS logit matrix, masking out padded visual tokens (zeros).
        All internal computation is done in float32 for numerical stability.
        """
        B = visual_tokens_all.shape[0]
        N_T = text_latents_all.shape[0]
        device = visual_tokens_all.device
        orig_dtype = visual_tokens_all.dtype

        # Cast to float32 for stable softmax/attention over thousands of tokens
        visual_tokens_f32 = visual_tokens_all.float()
        text_latents_f32 = text_latents_all.float()
        temp_f32 = temp.float() if isinstance(temp, torch.Tensor) else temp

        # NaN guard: replace any NaN/Inf in inputs with 0
        visual_tokens_f32 = torch.nan_to_num(visual_tokens_f32, nan=0.0, posinf=0.0, neginf=0.0)
        text_latents_f32 = torch.nan_to_num(text_latents_f32, nan=0.0, posinf=0.0, neginf=0.0)

        visual_tokens_norm = l2norm(visual_tokens_f32)
        logits = torch.zeros(N_T, B, device=device, dtype=torch.float32)

        for sent_start in range(0, N_T, chunk_size):
            sent_end = min(sent_start + chunk_size, N_T)
            chunk_text = text_latents_f32[sent_start:sent_end]

            for img_idx in range(B):
                img_tokens = visual_tokens_f32[img_idx]
                img_tokens_norm = visual_tokens_norm[img_idx]

                attn_scores = einsum('c d, n d -> c n', chunk_text, img_tokens_norm)

                # APPLY MASK: Set padded tokens to -inf so they have 0 attention weight
                mask = token_mask_all[img_idx] # [9216]
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0), -1e4)

                attn_weights = F.softmax(attn_scores, dim=-1)
                pooled = einsum('c n, n d -> c d', attn_weights, img_tokens)
                pooled_norm = l2norm(pooled)
                sim = einsum('c d, c d -> c', pooled_norm, chunk_text)
                logits[sent_start:sent_end, img_idx] = sim * temp_f32

        # Clamp logits to prevent extreme values from causing NaN in cross_entropy
        logits = logits.clamp(-100.0, 100.0)

        return logits.to(orig_dtype)

    # --- KEY LOGIC: Encode and Pad ---
    def _encode_visual_tokens(self, image, real_volume_mask, vis_proj_layer,
                            text_latents=None, num_sentences_per_image=1):
        """
        Encodes visual tokens and applies zero-padding to match MAX_TOKENS.
        Returns: (merged_tokens, token_mask)
        """
        b, r, c, d, h, w = image.shape
        merged = None

        # --- Fusion Logic ---
        if self.fusion_mode == "early":
            img_in = image[:, 0]
            enc = self.run_checkpoint(self.visual_transformer, img_in)
            merged = vis_proj_layer(enc)

        elif self.fusion_mode == "mid_cnn":
            flat_img = rearrange(image, 'b r c d h w -> (b r) c d h w')
            cnn_features = self.run_checkpoint(self.visual_transformer.forward_cnn, flat_img)
            cnn_features = rearrange(cnn_features, '(b r) t h w d -> b r t h w d', r=r)
            m = real_volume_mask.view(b, r, 1, 1, 1, 1).to(cnn_features.dtype)
            merged = (cnn_features * m).sum(1) / m.sum(1).clamp(min=1.0)
            enc = self.run_checkpoint(self.visual_transformer.forward_transformer, merged)
            merged = vis_proj_layer(enc)

        elif self.fusion_mode == "late":
            all_tokens = []
            for i in range(r):
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    print(f"[MRRATE rank=0] encoding volume {i+1}/{r}...", flush=True)
                enc = self.run_checkpoint(self.visual_transformer, image[:, i])
                all_tokens.append(vis_proj_layer(enc))
            all_tokens = torch.stack(all_tokens, dim=1)
            m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
            merged = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        elif self.fusion_mode == "late_attn":
            all_tokens = []
            for i in range(r):
                enc = self.run_checkpoint(self.visual_transformer, image[:, i])
                all_tokens.append(vis_proj_layer(enc))
            all_tokens = torch.stack(all_tokens, dim=1)

            if self.pooling_strategy == "simple_attn":
                merged = self.recon_pool(all_tokens, mask=real_volume_mask)
            elif self.pooling_strategy in ["cross_attn", "gated"]:
                # Logic to get text_query for pooling
                if text_latents is not None and text_latents.shape[0] == b * num_sentences_per_image:
                     text_query = text_latents.view(b, num_sentences_per_image, -1).mean(dim=1)
                elif text_latents is not None:
                     text_query = text_latents
                else: 
                     text_query = None

                if text_query is not None:
                    merged = self.recon_pool(all_tokens, text_query, mask=real_volume_mask)
                else:
                    m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
                    merged = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        # --- Padding Logic ---
        B, N, D = merged.shape
        # MAX_TOKENS: Set to accommodate largest possible output (128 frames * 144 spatial = 18432)
        MAX_TOKENS = 18432 
        
        token_mask = torch.ones((B, N), device=merged.device, dtype=torch.bool)
        
        if N < MAX_TOKENS:
            padding_size = MAX_TOKENS - N
            # Pad tokens with zero
            padding = torch.zeros((B, padding_size, D), device=merged.device, dtype=merged.dtype)
            merged = torch.cat([merged, padding], dim=1)
            
            # Pad mask with False
            mask_padding = torch.zeros((B, padding_size), device=merged.device, dtype=torch.bool)
            token_mask = torch.cat([token_mask, mask_padding], dim=1)
            
        return merged, token_mask

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        clean_state = {}
        for k, v in pt.items():
            if k.startswith("module."):
                clean_state[k[len("module."):]] = v
            else:
                clean_state[k] = v
        self.load_state_dict(clean_state, strict=False)

    def forward(
            self,
            text_input,
            image,
            device,
            real_volume_mask=None,
            num_sentences_per_image=1,
            sentence_mask=None,
            return_loss=False,
            aug_text=None,
            mode='clip',
            debug=False,
            **kwargs
    ):
        b, r, c, d, h, w = image.shape

        if debug and dist.is_initialized():
            print(f"[DEBUG MRRATE rank={dist.get_rank()}] forward start: image={image.shape}", flush=True)

        if real_volume_mask is None:
            real_volume_mask = torch.ones(b, r, dtype=torch.bool, device=device)

        use_extra_proj = self.extra_latent_projection and return_loss
        vis_proj_layer = self.to_visual_latent_extra if use_extra_proj else self.to_visual_latent
        text_proj_layer = self.to_text_latent_extra if use_extra_proj else self.to_text_latent

        # 1. Encode Text
        if debug and dist.is_initialized():
            print(f"[DEBUG MRRATE rank={dist.get_rank()}] encoding text...", flush=True)

        text_latents = None
        if text_input is not None:
            input_ids = text_input.input_ids
            attention_mask = text_input.attention_mask

            if self.use_gradient_checkpointing and self.training:
                text_output = checkpoint(self.text_transformer, input_ids, attention_mask, use_reentrant=False)
                enc_text = text_output.last_hidden_state if hasattr(text_output, 'last_hidden_state') else text_output[0]
            else:
                text_output = self.text_transformer(input_ids, attention_mask=attention_mask)
                enc_text = text_output.last_hidden_state

            text_latents = l2norm(text_proj_layer(enc_text[:, 0, :]))

        if debug and dist.is_initialized():
            print(f"[DEBUG MRRATE rank={dist.get_rank()}] text done, encoding visual...", flush=True)

        # 2. Encode Visual Tokens (Returns Tokens + Mask)
        visual_tokens, token_mask = self._encode_visual_tokens(
            image, real_volume_mask, vis_proj_layer,
            text_latents=text_latents,
            num_sentences_per_image=num_sentences_per_image
        )

        if debug and dist.is_initialized():
            print(f"[DEBUG MRRATE rank={dist.get_rank()}] visual done: tokens={visual_tokens.shape}, mask={token_mask.shape}", flush=True)

        # 3. Training Loss
        if return_loss:
            temp = self.logit_temperature.float().clamp(min=0.0, max=4.0).exp()

            if debug and dist.is_initialized():
                print(f"[DEBUG MRRATE rank={dist.get_rank()}] starting all_gather for text_latents...", flush=True)

            if dist.is_initialized():
                text_latents_all = all_gather_batch(text_latents)
                if debug:
                    print(f"[DEBUG MRRATE rank={dist.get_rank()}] text_latents gathered: {text_latents_all.shape}", flush=True)
                visual_tokens_all = all_gather_batch(visual_tokens)
                if debug:
                    print(f"[DEBUG MRRATE rank={dist.get_rank()}] visual_tokens gathered: {visual_tokens_all.shape}", flush=True)
                # GATHER TOKEN MASK
                token_mask_all = all_gather_batch(token_mask.float()).bool()
                if debug:
                    print(f"[DEBUG MRRATE rank={dist.get_rank()}] token_mask gathered: {token_mask_all.shape}", flush=True)

                if sentence_mask is not None:
                    mask_all = all_gather_batch(sentence_mask.view(-1).float()).bool()
                else:
                    mask_all = None
                if debug:
                    print(f"[DEBUG MRRATE rank={dist.get_rank()}] all_gather complete, computing logits...", flush=True)
            else:
                text_latents_all = text_latents
                visual_tokens_all = visual_tokens
                token_mask_all = token_mask
                mask_all = sentence_mask.view(-1) if sentence_mask is not None else None

            # Compute logits: local images vs global text
            # IMPORTANT: Use LOCAL token_mask, not global token_mask_all
            # Each GPU computes similarities for its own images against all texts
            local_logits = self._compute_logit_matrix(
                visual_tokens, text_latents_all, token_mask, temp, chunk_size=4
            )

            if dist.is_initialized():
                gathered_list = dist_nn_fun.all_gather(local_logits)
                logits = torch.cat(gathered_list, dim=1)
            else:
                logits = local_logits

            # Standard Contrastive Loss with sentence validity masking
            # Cast logits to float32 for numerically stable loss (cross_entropy, logsumexp)
            logits = logits.float()

            B_global = logits.shape[1]
            N_T = text_latents_all.shape[0]

            # Map every sentence to its image index
            sentence_to_image_map = torch.arange(B_global, device=device).repeat_interleave(num_sentences_per_image)
            image_membership = F.one_hot(sentence_to_image_map, num_classes=B_global).bool()

            if mask_all is not None:
                mask_float = mask_all.float()
                # Text-to-Image Loss
                loss_t_all = F.cross_entropy(logits, sentence_to_image_map, reduction='none')
                # Use torch.where to avoid 0 * NaN = NaN
                loss_t_all = torch.where(mask_all, loss_t_all, torch.zeros_like(loss_t_all))
                loss_t = loss_t_all.sum() / mask_float.sum().clamp(min=1.0)

                # Image-to-Text Loss
                pos_logits = logits[torch.arange(N_T, device=device), sentence_to_image_map]
                not_from_image = ~image_membership
                valid_sentences = mask_all.view(N_T, 1).expand(N_T, B_global)

                logits_masked_for_neg = logits.clone()
                logits_masked_for_neg[~(not_from_image & valid_sentences)] = -1e4

                logsumexp_neg_per_image = torch.logsumexp(logits_masked_for_neg, dim=0)
                logsumexp_neg = logsumexp_neg_per_image[sentence_to_image_map]

                denom_stack = torch.stack([pos_logits, logsumexp_neg], dim=1)
                log_denom = torch.logsumexp(denom_stack, dim=1)

                loss_i_all = -pos_logits + log_denom
                # Use torch.where to avoid 0 * NaN = NaN
                loss_i_all = torch.where(mask_all, loss_i_all, torch.zeros_like(loss_i_all))
                loss_i = loss_i_all.sum() / mask_float.sum().clamp(min=1.0)
            else:
                loss_t = F.cross_entropy(logits, sentence_to_image_map)

                pos_logits = logits[torch.arange(N_T, device=device), sentence_to_image_map]
                not_from_image = ~image_membership
                logits_masked_for_neg = logits.clone()
                logits_masked_for_neg[~not_from_image] = -1e4

                logsumexp_neg_per_image = torch.logsumexp(logits_masked_for_neg, dim=0)
                logsumexp_neg = logsumexp_neg_per_image[sentence_to_image_map]
                denom_stack = torch.stack([pos_logits, logsumexp_neg], dim=1)
                log_denom = torch.logsumexp(denom_stack, dim=1)
                loss_i = (-pos_logits + log_denom).mean()

            return loss_t + loss_i

        # 4. Inference
        else:
            # Simple masked pooling for inference
            mask_bc = token_mask.unsqueeze(-1)
            sum_tokens = (visual_tokens * mask_bc).sum(dim=1)
            count_tokens = mask_bc.sum(dim=1).clamp(min=1.0)
            return l2norm(sum_tokens / count_tokens)