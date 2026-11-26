"""MR writer model: the FORA report writer over the MR-RATE MIL token stream.

``fora.reportgen.writer.dino_stream.DinoCTReportGenerator`` (Gemma-4-31B-it, LoRA r16/a32 on the language model's
attention and MLP projections) with
  * visual prefix: every MIL token of every sequence (1304-d classifier input) -> QueryResampler 1304 -> 1024
    (depth 2, 8 heads) -> 512 latents -> LLM width, plus zero-initialised per-token embeddings of the sequence kind
    (T1, T2, FLAIR, SWI, ...) and the sequence ordinal, and one trainable study-type token (brain MRI);
  * one frozen-Platt label token per diagnosis;
  * the plan text (``plan.MRPlanBuilder``) whose planned findings carry the attention-pooled feature and up to 16
    focus tokens (``evidence``), each 1304 + 16 auxiliary dims.
No zoomed evidence images: the Gemma vision tower is removed.
"""
from __future__ import annotations

import torch

from .evidence import AUX_DIM, EvidenceGeometry, finding_evidence
from .mil import MAX_SERIES
from .plan import MRPlanBuilder, sequences_text

STUDY_TYPE_BRAIN_MRI = 0
VISUAL_DIM = 1304
FEATURE_DIM = VISUAL_DIM + AUX_DIM
N_VIEW_KINDS = 12            # mr_dino.geometry.SEQUENCE_KINDS


def load_llm(llm_path: str, device, lora_r: int, lora_alpha: int):
    """Gemma-4-31B-it + text LoRA (vision and audio towers removed)."""
    from fora.reportgen.plan import diagnostic_plan
    from fora.reportgen.writer.llm import build_gemma_lora
    llm, tokenizer, llm_dim = build_gemma_lora(llm_path, device, lora_r=lora_r, lora_alpha=lora_alpha, fimg_tokens=0)
    diagnostic_plan.set_finding_placeholder(llm.fora_llm_family)
    return llm, tokenizer, llm_dim


def build_report_model(llm, tokenizer, llm_dim, labels, plan_builder: MRPlanBuilder, device, args):
    from fora.reportgen.plan import diagnostic_plan as DP
    from fora.reportgen.writer.conditioning import semantic_label_embeddings
    from fora.reportgen.writer.dino_stream import DinoCTReportGenerator
    semantics = semantic_label_embeddings(llm, tokenizer, plan_builder.names, device)
    placeholder = -1
    if args.finding_tokens:
        placeholder = int(tokenizer.convert_tokens_to_ids(DP.FINDING_PLACEHOLDER))
        probe = tokenizer(DP.FINDING_PLACEHOLDER, add_special_tokens=False).input_ids
        if probe != [placeholder] or placeholder in (tokenizer.unk_token_id, None):
            raise RuntimeError(f"{DP.FINDING_PLACEHOLDER} is not a single reserved token: {probe}")
    C = len(labels)
    return DinoCTReportGenerator(
        llm, resampler_dim=args.resampler_dim, n_study_types=args.study_types, visual_dim=VISUAL_DIM,
        llm_dim=llm_dim, num_visual_tokens=args.num_visual_tokens, resampler_depth=args.resampler_depth,
        resampler_heads=args.resampler_heads, n_view_kinds=N_VIEW_KINDS, max_series=MAX_SERIES,
        semantic_embeddings=semantics, eligible_class_indices=torch.arange(C, device=device),
        calibration_slope=plan_builder.slope.to(device), calibration_intercept=plan_builder.intercept.to(device),
        diagnostic_consistency_classes=C, diagnostic_consistency_weight=0.0,
        finding_feature_dim=FEATURE_DIM if args.finding_tokens else 0, finding_placeholder_id=placeholder,
        fimg_tokens=0,
    ).to(device, dtype=torch.bfloat16)


def plan_inputs(plan_builder: MRPlanBuilder, logits: torch.Tensor, enc, geometry: EvidenceGeometry, labels,
                device, finding_tokens: bool = True):
    """Plan ids and the evidence features of the planned findings in placeholder order.

    ``logits`` [1, C] are the study's precomputed MIL logits (the calibration source); ``enc`` is the study encoded
    by ``mil.MRVisualStream``."""
    selection = plan_builder.plan_selection(logits)
    planned = [*selection.confident, *selection.possible]
    descriptions, counts, rows = {-1: sequences_text(enc.kinds)}, {}, {}
    for c in planned:
        ev = finding_evidence(enc, int(c), labels[int(c)], geometry)
        if ev["description"]:
            descriptions[int(c)] = ev["description"]
        if finding_tokens:
            counts[int(c)] = int(ev["tokens"].shape[0])
            rows[int(c)] = ev["tokens"]
    plan = plan_builder.plan_ids(logits, device, location_descriptions=descriptions,
                                 finding_token_classes=counts if finding_tokens else None)
    features = None
    if finding_tokens:
        order = plan_builder.planned_finding_order(logits, counts)
        features = (torch.cat([rows[c] for c in order]).to(device) if order
                    else torch.zeros(0, FEATURE_DIM, device=device))
    return plan, features, selection
