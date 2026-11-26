#!/usr/bin/env python3
"""MR writer, offline DPO stage (the FORA preference recipe).

Policy and frozen reference = the SFT checkpoint (``fora_report.writer.fsdp_dpo.FSDPReferenceWeights``; the start must
equal the reference exactly). Studies = TRAIN studies with at least one pair (``dpo_pairs``; the chosen text is the
study's own findings, re-checked by sha256). Per visit at most K = 2 pairs. Objective per paired study:
    dpo_weight * mean_k -log sigmoid(beta * margin_k) + chosen_ce_weight * token-mean CE(chosen),   beta = 0.1
with the sequence score = sum log p of 'Findings:\\n' + text + EOS. A study without pairs on every rank of its FSDP
shard group takes a plain CE step. Defaults: connector lr 2e-5, LoRA lr 5e-6, warm-up 2 %, cosine over --max_steps,
step_NNNNNN.pt every 100 steps (latest.pt = hard link), exact resume with --resume.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from . import train_sft as S
from .dpo_pairs import SCHEMA

OBJECTIVE = "offline_dpo_sum_logprob"
PREFERENCE_DEFAULTS = dict(connector_lr=2e-5, llm_lr=5e-6, warmup_fraction=0.02, save_every_steps=100)
CHECKED_KEYS = ("bundle", "dense", "lora_r", "lora_alpha", "num_visual_tokens", "resampler_depth", "resampler_heads",
                "resampler_dim", "study_types", "finding_tokens", "max_report_tokens")


class PairStore:
    def __init__(self, path: str):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.summary = json.loads(self.con.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
        if self.summary.get("schema") != SCHEMA:
            raise ValueError(f"not an MR preference store: {path}")

    def pairs(self, uid: str):
        return self.con.execute("SELECT pair_id, pair_type, attribute_type, chosen_sha256, rejected_text FROM pairs "
                                "WHERE report_uid=? ORDER BY pair_id", (str(uid),)).fetchall()

    def uids(self) -> set:
        return {r[0] for r in self.con.execute("SELECT DISTINCT report_uid FROM pairs")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_common_arguments(p)
    p.add_argument("--dpo_pairs", required=True)
    p.add_argument("--sft", required=True, help="start checkpoint = warm start = frozen reference")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_weight", type=float, default=1.0)
    p.add_argument("--dpo_chosen_ce_weight", type=float, default=1.0)
    p.add_argument("--dpo_unpaired_ce_weight", type=float, default=1.0)
    p.add_argument("--dpo_pairs_per_visit", type=int, default=2)
    p.add_argument("--dpo_mix", type=float, default=1.0)
    p.add_argument("--dpo_recompute_tolerance", type=float, default=0.5)
    p.add_argument("--dpo_recompute_abort", type=float, default=20.0)
    p.add_argument("--log_every_steps", type=int, default=5)
    p.set_defaults(**PREFERENCE_DEFAULTS)
    return p.parse_args(argv)


def main(argv=None):
    from fora.reportgen.cxr.model import cosine_scheduler, split_parameters, target_ids
    from fora_report.ct.dpo_loss import DPOStatistics, disable_preference_dropout
    from fora_report.cxr.data import epoch_schedule
    from fora_report.cxr.train_dpo import dpo_step
    from fora_report.cxr.train_sft import StopFlag, init_distributed
    from fora_report.writer import fsdp as reportgen_fsdp
    from fora_report.writer import fsdp_dpo as reportgen_fsdp_dpo

    from .data import DenseLogits, Prefetcher
    from .mil import MRVisualStream, add_bundle_paths
    from .model import STUDY_TYPE_BRAIN_MRI, plan_inputs
    args = parse_args(argv)
    add_bundle_paths(args.bundle)
    rank, world, local, local_world, device = init_distributed(args.pg_timeout_s)
    main_rank = rank == 0
    results = Path(args.results_dir)
    if main_rank:
        results.mkdir(parents=True, exist_ok=True)

    def log(**kw):
        if main_rank:
            print(json.dumps(kw), flush=True)

    sft = torch.load(args.sft, map_location="cpu", weights_only=False, mmap=True)
    sft_meta = sft["metadata"]
    sft_args = sft_meta["arguments"]
    if Path(sft_args["calibration"]).resolve() != Path(args.calibration).resolve():
        raise RuntimeError(f"start calibration {sft_args['calibration']} != preference calibration {args.calibration}")
    for key in CHECKED_KEYS:
        if str(sft_args[key]) != str(getattr(args, key)):
            raise RuntimeError(f"{key}: start checkpoint {sft_args[key]} != {getattr(args, key)}")
    reference_state = sft["trainable_model"]

    store = PairStore(args.dpo_pairs)
    paired = store.uids()
    calibration = json.loads(Path(args.calibration).read_text())
    labels = list(calibration["labels"])
    dense = DenseLogits(args.dense, labels, calibration)
    cases, _sizes, reports, positives_of = S.training_data(args, labels, log)
    cases = [c for c in cases if c["study_uid"] in paired and c["study_uid"] in dense]
    sizes = np.array([len(c["image_paths"]) for c in cases], np.int64)
    steps_per_epoch = max(1, len(cases) // world)
    total_steps = args.max_steps
    log(event="data", paired_cases=len(cases), world=world, max_steps=total_steps, pairs=store.summary["pairs_by_type"])

    stream = MRVisualStream(args.bundle, device)
    geometry = S.evidence_geometry(stream)
    llm, tokenizer, plan_builder, model, mesh = S.setup_writer(args, labels, device, world, local_world, log)
    missing = reportgen_fsdp_dpo.load_exact_warm_start(model, reference_state)
    if missing:
        raise RuntimeError(f"start checkpoint lacks trainable tensors: {missing[:5]}")
    dropout_changes = disable_preference_dropout(model)
    reference = reportgen_fsdp_dpo.FSDPReferenceWeights(model, reference_state, device)
    del reference_state, sft
    slot_group = reportgen_fsdp_dpo.slot_group(mesh)
    adapter = reportgen_fsdp.FSDPSyncAdapter(model)
    connector, adapters = split_parameters(model)
    optimizer = torch.optim.AdamW([{"params": connector, "lr": args.connector_lr},
                                   {"params": adapters, "lr": args.llm_lr}],
                                  weight_decay=args.weight_decay, foreach=True)
    scheduler = cosine_scheduler(optimizer, total_steps, args.warmup_fraction)
    schedule_key = dict(paired_cases=len(cases), world=world, max_steps=total_steps, seed=args.seed, bucket=args.bucket,
                        pairs=str(Path(args.dpo_pairs).resolve()), sft=str(Path(args.sft).resolve()))
    start = 0
    if args.resume:
        package = torch.load(args.resume, map_location="cpu", weights_only=False)
        if package["metadata"]["dpo"]["schedule"] != schedule_key:
            raise RuntimeError(f"resume schedule differs: {package['metadata']['dpo']['schedule']}")
        reportgen_fsdp.load_full_trainable_state(model, package["trainable_model"])
        reportgen_fsdp.load_full_optimizer_state(optimizer, package["optimizer"])
        scheduler.load_state_dict(package["scheduler"])
        start = int(package["optimizer_step"])
        del package
        log(event="resumed", path=args.resume, optimizer_step=start)
    else:
        diff = reference.max_abs_difference()
        if diff != 0.0:
            raise RuntimeError(f"preference training must start exactly from the reference: max|diff|={diff:g}")
    log(event="dpo", beta=args.dpo_beta, k=args.dpo_pairs_per_visit, lr=[args.connector_lr, args.llm_lr],
        dropout_disabled=len(dropout_changes), memory=reportgen_fsdp.describe_memory(device))

    metadata = json.loads(json.dumps(sft_meta, default=str))
    metadata["arguments"] = {**metadata["arguments"], **{k: v for k, v in vars(args).items() if k.startswith("dpo_")},
                             "connector_lr": args.connector_lr, "llm_lr": args.llm_lr}
    metadata["dpo"] = dict(schedule=schedule_key, objective=OBJECTIVE, pairs_summary=store.summary,
                           warmup_fraction=args.warmup_fraction)

    def save(step_done):
        t0 = time.time()
        state = reportgen_fsdp.full_trainable_state(model, main_rank)
        opt = reportgen_fsdp.full_optimizer_state(optimizer, main_rank)
        name = f"step_{step_done:06d}.pt"
        if main_rank:
            package = dict(trainable_model=state, optimizer=opt, scheduler=scheduler.state_dict(), epoch=0,
                           next_step=step_done, optimizer_step=step_done, metadata=metadata)
            tmp = results / f"{name}.tmp"
            torch.save(package, tmp)
            tmp.replace(results / name)
            lnk = results / "latest.pt.lnk"
            lnk.unlink(missing_ok=True)
            os.link(results / name, lnk)
            lnk.replace(results / "latest.pt")
        dist.barrier(device_ids=[local])
        log(event="saved", file=name, optimizer_step=step_done, seconds=round(time.time() - t0, 1))

    stopper = StopFlag(device)
    model.train()
    stats = DPOStatistics(device)
    study_types = torch.tensor([STUDY_TYPE_BRAIN_MRI], dtype=torch.long, device=device)
    modality_empty = torch.empty((0, args.resampler_dim), dtype=torch.bfloat16, device=device)
    schedule = np.concatenate([epoch_schedule(len(cases), sizes, world, e, args.seed, args.bucket)
                               for e in range(-(-total_steps // steps_per_epoch))])[:total_steps]
    order = schedule[start:, rank]
    loader = Prefetcher(stream, cases, order, reports, depth=args.loader_depth, workers=args.loader_workers)
    max_tok = args.max_report_tokens
    window, launched = [], 0
    for k in range(len(order)):
        step = start + k
        t0 = time.time()
        case, volumes, text = loader.get(k)
        enc = stream.encode(case, volumes)
        del volumes
        uid = case["study_uid"]
        logits = dense.get(uid, device)
        plan, features, _ = plan_inputs(plan_builder, logits, enc, geometry, labels, device, args.finding_tokens)
        t_enc = time.time() - t0
        positives = torch.as_tensor(positives_of.get(uid, np.zeros(0, np.int64)), dtype=torch.long, device=device)
        consistency = plan_builder.consistency_targets(positives, device)
        chosen = target_ids(tokenizer, text, max_tok + 1, device)

        def forward(target):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return adapter(enc.tokens, modality_empty, plan, target, positives, logits, plan, consistency,
                               finding_features=features, dino_view_kind=enc.view_kind,
                               dino_view_series=enc.view_series, dino_study_types=study_types)

        def score(kind, target):
            out = forward(target)
            if not torch.isclose(out["loss"].detach().float(), out["report_loss"].float(), rtol=1e-4, atol=1e-5):
                raise RuntimeError("preference training needs loss == token-mean report NLL")
            return -out["loss"].float() * target.numel(), out

        def sync(is_last):
            return contextlib.nullcontext() if is_last else adapter.no_sync()

        env = SimpleNamespace(args=args, step=step, rank=rank, uid=uid, text=text, chosen=chosen, max_tok=max_tok,
                              tokenizer=tokenizer, device=device, store=store, stats=stats, reference=reference,
                              slot_group=slot_group, forward=forward, score=score, sync=sync,
                              steps_per_epoch=steps_per_epoch)
        value, n_pairs = dpo_step(env)
        grad_norm = reportgen_fsdp.clip_grad_norm([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        done = step + 1
        launched += 1
        st = torch.tensor([value, float(n_pairs), time.time() - t0, t_enc,
                           torch.cuda.max_memory_allocated(device) / 2**30], device=device)
        mean = st.clone(); dist.all_reduce(mean); mean /= world
        peak = st.clone(); dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        window.append(mean[2].item())
        if done <= 10 or done % args.log_every_steps == 0:
            line = stats.line(done, lambda t: dist.all_reduce(t))
            log(event="step", opt_step=done, loss=round(mean[0].item(), 4), pairs_per_study=round(mean[1].item(), 2),
                s_step=round(mean[2].item(), 2), s_step_max=round(peak[2].item(), 2), s_enc=round(mean[3].item(), 2),
                s_step_avg10=round(float(np.mean(window[-10:])), 2), grad_norm=round(float(grad_norm), 3),
                lr=[g["lr"] for g in optimizer.param_groups], peak_gib=round(peak[4].item(), 1))
            if main_rank:
                print(line, flush=True)
        del enc, features, plan, chosen, env
        stop = stopper.requested() or bool(args.stop_after_steps and launched >= args.stop_after_steps)
        if done == total_steps or (args.save_every_steps and done % args.save_every_steps == 0) or stop:
            save(done)
        if stop and done < total_steps:
            loader.close()
            log(event="stopped", optimizer_step=done)
            dist.destroy_process_group()
            return 0
    loader.close()
    if main_rank:
        (results / "COMPLETE.json").write_text(json.dumps(dict(max_steps=total_steps, time=time.time())))
    log(event="complete", optimizer_step=total_steps)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
