from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from .targets import ReportTarget


class CrossAttentionBlock(nn.Module):
    """Exact all-token attention with a bounded-memory streaming fallback."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("resampler dimension must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_k = nn.Linear(dim, dim, bias=False)
        self.cross_v = nn.Linear(dim, dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def _stream_step(
        self,
        running_max: torch.Tensor,
        denominator: torch.Tensor,
        numerator: torch.Tensor,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.memory_norm(memory)
        key = self.cross_k(normalized).view(
            1, -1, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.cross_v(normalized).view(
            1, -1, self.heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores.mul(self.head_dim**-0.5).float()
        chunk_max = scores.amax(dim=-1).detach()
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max.unsqueeze(-1))
        denominator = denominator * old_scale + weights.sum(dim=-1)
        numerator = (
            numerator * old_scale.unsqueeze(-1)
            + torch.matmul(weights, value.float())
        )
        return new_max, denominator, numerator

    def _stream(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        shape = (1, self.heads, query.shape[2])
        running_max = torch.full(
            shape, -torch.inf, dtype=torch.float32, device=query.device
        )
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros(
            *shape, self.head_dim, dtype=torch.float32, device=query.device
        )
        chunk_size = int(os.environ.get("MRRATE_REPORT_STREAM_CHUNK", "8192"))
        if chunk_size <= 0:
            raise ValueError("MRRATE_REPORT_STREAM_CHUNK must be positive")
        for start in range(0, memory.shape[1], chunk_size):
            chunk = memory[:, start : start + chunk_size]
            if self.training and torch.is_grad_enabled():
                running_max, denominator, numerator = checkpoint(
                    self._stream_step,
                    running_max,
                    denominator,
                    numerator,
                    query,
                    chunk,
                    use_reentrant=False,
                )
            else:
                running_max, denominator, numerator = self._stream_step(
                    running_max, denominator, numerator, query, chunk
                )
        return (numerator / denominator.clamp_min(1e-20).unsqueeze(-1)).to(
            query.dtype
        )

    def forward(self, latents: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(latents)
        latents = latents + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        query = self.cross_q(self.cross_norm(latents)).view(
            1, -1, self.heads, self.head_dim
        ).transpose(1, 2)
        threshold = int(os.environ.get("MRRATE_REPORT_STREAM_THRESHOLD", "131072"))
        if threshold > 0 and memory.shape[1] > threshold:
            attended = self._stream(query, memory)
        else:
            normalized_memory = self.memory_norm(memory)
            key = self.cross_k(normalized_memory).view(
                1, -1, self.heads, self.head_dim
            ).transpose(1, 2)
            value = self.cross_v(normalized_memory).view(
                1, -1, self.heads, self.head_dim
            ).transpose(1, 2)
            attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(1, -1, self.dim)
        latents = latents + self.cross_out(attended)
        return latents + self.mlp(self.mlp_norm(latents))


class QueryResampler(nn.Module):
    def __init__(self, dim: int, num_queries: int, depth: int, heads: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(dim, heads) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or not tokens.shape[0]:
            raise ValueError("tokens must be a non-empty [N,D] tensor")
        memory = tokens.unsqueeze(0)
        latents = self.latents.to(memory.dtype)
        for block in self.blocks:
            latents = (
                checkpoint(block, latents, memory, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else block(latents, memory)
            )
        return self.output_norm(latents)


def activate_adapter(model: nn.Module, name: str) -> None:
    model.set_adapter(name)
    # PEFT freezes the inactive adapter during set_adapter. Both adapters are
    # used before one backward pass, so keep both optimizer-visible.
    for parameter_name, parameter in model.named_parameters():
        if "lora_" in parameter_name:
            parameter.requires_grad_(True)


def build_gemma_writer(
    model_path: str,
    device: torch.device,
    *,
    lora_r: int,
    lora_alpha: int,
) -> tuple[nn.Module, AutoTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    llm.model.vision_tower = None
    llm.model.multi_modal_projector = None
    llm.requires_grad_(False)
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm = get_peft_model(llm, lora, adapter_name="report")
    activate_adapter(llm, "report")
    llm.to(device)
    return llm, tokenizer, int(llm.config.text_config.hidden_size)


@torch.no_grad()
def label_semantic_embeddings(
    llm: nn.Module, tokenizer: AutoTokenizer, names: Sequence[str]
) -> torch.Tensor:
    embedding = llm.get_input_embeddings()
    vectors = []
    device = embedding.weight.device
    for name in names:
        ids = tokenizer(
            f"MRI finding: {name}", add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        vectors.append(embedding(ids).float().mean(dim=1).squeeze(0))
    return torch.stack(vectors)


class ReportWriter(nn.Module):
    REPORT_PROMPT = (
        "Write the complete MRI findings supported by the visual evidence. "
        "Preserve both positive findings and explicit negative findings. "
        "Do not invent location details. If the source report has no findings, "
        "write <NONE>.\nFindings:"
    )

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        label_embeddings: torch.Tensor | None,
        *,
        visual_dim: int = 512,
        num_visual_queries: int = 512,
        resampler_depth: int = 2,
        resampler_heads: int = 8,
        max_target_tokens: int = 384,
        mil_conditioning: str = "all_classes",
        llm_dim: int | None = None,
    ) -> None:
        super().__init__()
        if mil_conditioning not in ("all_classes", "none"):
            raise ValueError(f"Unknown mil_conditioning: {mil_conditioning!r}")
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_target_tokens = int(max_target_tokens)
        self.mil_conditioning = mil_conditioning
        if mil_conditioning == "all_classes":
            if label_embeddings is None:
                raise ValueError("all_classes conditioning requires label embeddings")
            llm_dim = int(label_embeddings.shape[1])
        else:
            if label_embeddings is not None:
                raise ValueError(
                    "mil_conditioning=none must not receive label embeddings"
                )
            if not llm_dim:
                raise ValueError("mil_conditioning=none requires an explicit llm_dim")
            llm_dim = int(llm_dim)
        self.resampler = QueryResampler(
            visual_dim, num_visual_queries, resampler_depth, resampler_heads
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, llm_dim)
        )
        self.image_start = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
        self.image_end = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
        if mil_conditioning == "all_classes":
            self.register_buffer(
                "label_embeddings", label_embeddings.float(), persistent=True
            )
            self.mil_value_projection = nn.Sequential(
                nn.Linear(2, 64), nn.GELU(), nn.Linear(64, llm_dim)
            )
            self.mil_norm = nn.LayerNorm(llm_dim)

    def shared_prefix(
        self,
        tokens: torch.Tensor,
        mil_logits: torch.Tensor | None,
        thresholds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.to(self.image_start.dtype)
        visual = self.visual_projection(self.resampler(tokens))
        visual_prefix = torch.cat((self.image_start, visual, self.image_end), dim=1)
        if self.mil_conditioning == "none":
            # No-classification-labels ablation: the writer sees visual
            # evidence only. Passing MIL inputs here indicates a mode mixup.
            if mil_logits is not None or thresholds is not None:
                raise ValueError(
                    "mil_conditioning=none writer received MIL conditioning"
                )
            return visual_prefix, visual_prefix.new_zeros(
                (1, 0, visual_prefix.shape[-1])
            )
        if mil_logits is None or thresholds is None:
            raise ValueError("all_classes conditioning requires MIL logits")
        probability = mil_logits.sigmoid().reshape(-1)
        thresholds = thresholds.to(probability).reshape(-1)
        if probability.numel() != self.label_embeddings.shape[0]:
            raise ValueError("MIL logits and label semantics differ")
        values = torch.stack((probability, probability - thresholds), dim=-1)
        mil_tokens = self.mil_norm(
            self.label_embeddings.to(values) + self.mil_value_projection(values)
        ).unsqueeze(0).to(visual_prefix.dtype)
        return visual_prefix, mil_tokens

    def _token_ids(self, text: str, *, append_eos: bool) -> torch.Tensor:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
            return_tensors="pt",
        ).input_ids[0]
        if append_eos:
            eos = torch.tensor([self.tokenizer.eos_token_id], dtype=ids.dtype)
            ids = torch.cat((ids, eos))
        return ids.to(self.image_start.device)

    def _report_loss(
        self, prefix: torch.Tensor, prompt: str, target: str
    ) -> torch.Tensor:
        activate_adapter(self.llm, "report")
        prompt_ids = self._token_ids(prompt, append_eos=False)
        target_ids = self._token_ids(target, append_eos=True)
        embedding = self.llm.get_input_embeddings()
        inputs = torch.cat(
            (
                prefix,
                embedding(prompt_ids).unsqueeze(0),
                embedding(target_ids).unsqueeze(0),
            ),
            dim=1,
        )
        attention_mask = torch.ones(
            1, inputs.shape[1], dtype=torch.long, device=inputs.device
        )
        token_type_ids = torch.zeros_like(attention_mask)
        token_type_ids[:, : prefix.shape[1]] = 1
        outputs = self.llm(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=False,
            logits_to_keep=target_ids.numel() + 1,
        )
        logits = outputs.logits[:, :-1].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        mil_logits: torch.Tensor | None,
        thresholds: torch.Tensor | None,
        *,
        max_new_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Greedy findings decoding from the same prefix used during training.

        The KV cache is used when the language model returns one; duck-typed
        test models without a cache fall back to full-sequence recomputation.
        ``use_cache=False`` forces the full-recompute path (each step is then
        numerically identical to a truncated training forward), which is
        useful to cross-check cached decoding.
        """

        if self.training:
            raise RuntimeError("generate() requires eval mode")
        activate_adapter(self.llm, "report")
        visual_prefix, mil_tokens = self.shared_prefix(tokens, mil_logits, thresholds)
        prompt_ids = self._token_ids(self.REPORT_PROMPT, append_eos=False)
        embedding = self.llm.get_input_embeddings()
        prefix = torch.cat(
            (visual_prefix, mil_tokens, embedding(prompt_ids).unsqueeze(0)), dim=1
        )
        conditioning_length = visual_prefix.shape[1] + mil_tokens.shape[1]
        limit = int(max_new_tokens) if max_new_tokens else self.max_target_tokens
        if limit <= 0:
            raise ValueError("max_new_tokens must be positive")
        stop_ids = {int(self.tokenizer.eos_token_id)}
        # Gemma3-IT declares two stop ids (<eos> and <end_of_turn>); the base
        # weights favor <end_of_turn>, so honor both. Training never targets
        # <end_of_turn>, which makes the extra stop loss-free.
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is not None:
            end_of_turn = convert("<end_of_turn>")
            if isinstance(end_of_turn, int) and end_of_turn >= 0:
                stop_ids.add(end_of_turn)
        generated: list[int] = []
        inputs = prefix
        past = None
        for _ in range(limit):
            length = prefix.shape[1] + len(generated)
            attention_mask = torch.ones(
                1, length, dtype=torch.long, device=inputs.device
            )
            token_type_ids = torch.zeros(
                1, inputs.shape[1], dtype=torch.long, device=inputs.device
            )
            if inputs.shape[1] == length:
                token_type_ids[:, :conditioning_length] = 1
            keyword_arguments = dict(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_cache=bool(use_cache),
                logits_to_keep=1,
            )
            if past is not None:
                keyword_arguments["past_key_values"] = past
            outputs = self.llm(**keyword_arguments)
            next_id = int(outputs.logits[0, -1].float().argmax())
            if next_id in stop_ids:
                break
            generated.append(next_id)
            next_ids = torch.tensor(
                [[next_id]], dtype=torch.long, device=inputs.device
            )
            past = (
                getattr(outputs, "past_key_values", None) if use_cache else None
            )
            next_embedding = embedding(next_ids[0]).unsqueeze(0)
            if past is None:
                inputs = torch.cat((inputs, next_embedding), dim=1)
            else:
                inputs = next_embedding
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def forward(
        self,
        tokens: torch.Tensor,
        mil_logits: torch.Tensor | None,
        thresholds: torch.Tensor | None,
        target: ReportTarget,
        *,
        loss_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        visual_prefix, mil_tokens = self.shared_prefix(
            tokens, mil_logits, thresholds
        )
        report_loss = self._report_loss(
            torch.cat((visual_prefix, mil_tokens), dim=1),
            self.REPORT_PROMPT,
            target.text,
        )
        return {"loss": report_loss * float(loss_scale), "report_loss": report_loss}


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, value in module.named_parameters() if value.requires_grad}
    # label_embeddings are deterministic but retaining them makes class-schema
    # drift immediately visible during resume. Absent under the
    # mil_conditioning=none ablation.
    if "label_embeddings" in module.state_dict():
        trainable.add("label_embeddings")
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if name in trainable
    }
