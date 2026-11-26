#!/usr/bin/env python3
"""MR writer, supervised fine-tuning (the FORA writer recipe on the MR-RATE MIL).

Gemma-4-31B-it + LoRA r16/a32 under FSDP2/HSDP (``fora_report.writer.fsdp``), trained on the frozen MIL token stream
of every study (``mil.MRVisualStream``, computed online). One study per GPU per optimizer step; global batch = world
size. Target: 'Findings:\\n' + the report's findings + EOS (at most 2,048 tokens). Plan from the precomputed study
logits (``calibrate``) with evidence descriptions and pooled + focus evidence tokens. Resampler 512 latents (depth 2,
8 heads, width 1024), connector lr 1e-4, LoRA lr 2e-5, AdamW wd 0.05, cosine with 5 % warm-up, grad-norm clip 1.0,
all auxiliary loss weights 0, LoRA dropout off, Gemma gradient checkpointing on, seed 42.

Checkpoints: latest.pt every --save_every_steps / --save_every_minutes and on SIGUSR1/SIGTERM (full trainable tensors
on CPU + plain optimizer state), checkpoint_epoch00E.pt at each epoch end. Exact resume with --resume.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def add_common_arguments(p: argparse.ArgumentParser) -> None:
    """Arguments shared by SFT, DPO and generation."""
    p.add_argument("--bundle", required=True, help="MIL bundle (backbone_release/, checkpoint/, code/, dfw_mil/)")
    p.add_argument("--data_folder", required=True, help="MR-RATE mri_atlas root")
    p.add_argument("--splits_csv", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--reports", required=True, help="report CSV glob (study_uid, findings)")
    p.add_argument("--calibration", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--llm_path", required=True, help="local snapshot of google/gemma-4-31B-it")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--connector_lr", type=float, default=1e-4)
    p.add_argument("--llm_lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_report_tokens", type=int, default=2048)
    p.add_argument("--num_visual_tokens", type=int, default=512)
    p.add_argument("--resampler_depth", type=int, default=2)
    p.add_argument("--resampler_heads", type=int, default=8)
    p.add_argument("--resampler_dim", type=int, default=1024)
    p.add_argument("--study_types", type=int, default=32)
    p.add_argument("--finding_tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bucket", type=int, default=8)
    p.add_argument("--save_every_steps", type=int, default=100)
    p.add_argument("--save_every_minutes", type=float, default=30)
    p.add_argument("--stop_after_steps", type=int, default=0, help="smoke test: stop (and save latest.pt) after N steps")
    p.add_argument("--limit_train", type=int, default=0, help="smoke test: use only the first N train studies")
    p.add_argument("--resume", default="")
    p.add_argument("--loader_depth", type=int, default=3)
    p.add_argument("--loader_workers", type=int, default=3)
    p.add_argument("--pg_timeout_s", type=int, default=36000)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(p)
    return p.parse_args(argv)


def evidence_geometry(stream):
    from .evidence import EvidenceGeometry
    from .regions import build_region_map
    args = stream.meta["args"]
    regions = build_region_map(tuple(args["target_shape"]), tuple(args["target_spacing"]), args["posterior_shift_mm"],
                               stream.patch)
    return EvidenceGeometry(stream.grid, stream.patch, stream.spacing, regions)


def training_data(args, labels, log):
    """Train cases with a report and dense logits, their sizes (sequence count) and the label positives."""
    from .data import read_cases, read_positives, read_reports
    reports = read_reports(args.reports)
    cases = [c for c in read_cases(args.data_folder, args.splits_csv, ("train",), "atlas_space")
             if c["study_uid"] in reports]
    if args.limit_train:
        cases = cases[: args.limit_train]
    sizes = np.array([len(c["image_paths"]) for c in cases], np.int64)
    log(event="data", train_cases=len(cases), reports=len(reports))
    return cases, sizes, reports, read_positives(args.labels_csv, labels)


def setup_writer(args, labels, device, world, local_world, log):
    """Gemma + LoRA, plan builder, writer, FSDP sharding."""
    from fora.reportgen.writer.llm import language_layers
    from fora_report.writer import fsdp as reportgen_fsdp

    from .model import build_report_model, load_llm
    from .plan import MRPlanBuilder
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    llm, tokenizer, llm_dim = load_llm(args.llm_path, device, args.lora_r, args.lora_alpha)
    if not reportgen_fsdp.fsdp_requested(llm.fora_llm_family):
        raise RuntimeError("the MR writer supports the FSDP (Gemma-4) path only")
    report = reportgen_fsdp.train_mode_for_checkpointing(llm, False)
    if not bool(llm.get_base_model().is_gradient_checkpointing):
        raise RuntimeError("Gemma gradient checkpointing is not active after PEFT wrapping")
    log(event="llm", family=llm.fora_llm_family, hidden=llm_dim, train_mode=report)
    plan_builder = MRPlanBuilder(args.calibration, labels, tokenizer)
    model = build_report_model(llm, tokenizer, llm_dim, labels, plan_builder, device, args)
    mesh = reportgen_fsdp.build_mesh("cuda", world, local_world)
    reportgen_fsdp.shard_report_model(model, language_layers(llm), mesh)
    torch.cuda.empty_cache()
    log(event="fsdp", mesh=list(mesh.shape), memory=reportgen_fsdp.describe_memory(device))
    return llm, tokenizer, plan_builder, model, mesh


def main(argv=None):
    from fora.reportgen.cxr.model import cosine_scheduler, split_parameters, target_ids
    from fora_report.cxr.data import epoch_schedule
    from fora_report.cxr.train_sft import StopFlag, init_distributed
    from fora_report.writer import fsdp as reportgen_fsdp

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

    calibration = json.loads(Path(args.calibration).read_text())
    labels = list(calibration["labels"])
    dense = DenseLogits(args.dense, labels, calibration)
    cases, sizes, reports, positives_of = training_data(args, labels, log)
    cases = [c for c in cases if c["study_uid"] in dense]
    sizes = np.array([len(c["image_paths"]) for c in cases], np.int64)
    steps_per_epoch = math.ceil(len(cases) / world)
    total_steps = args.epochs * steps_per_epoch
    log(event="schedule", train_cases=len(cases), world=world, steps_per_epoch=steps_per_epoch, total_steps=total_steps)

    stream = MRVisualStream(args.bundle, device)
    if stream.classes != labels:
        raise RuntimeError("MIL checkpoint classes != calibration labels")
    geometry = evidence_geometry(stream)
    llm, tokenizer, plan_builder, model, _mesh = setup_writer(args, labels, device, world, local_world, log)

    connector, adapters = split_parameters(model)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name.startswith("llm.") and "lora_" not in name:
            raise RuntimeError(f"unexpected trainable base parameter {name}")
    optimizer = torch.optim.AdamW([{"params": connector, "lr": args.connector_lr},
                                   {"params": adapters, "lr": args.llm_lr}],
                                  weight_decay=args.weight_decay, foreach=True)
    scheduler = cosine_scheduler(optimizer, total_steps, args.warmup_fraction)
    log(event="params", connector=sum(p.numel() for p in connector), lora=sum(p.numel() for p in adapters))

    schedule_key = dict(train_cases=len(cases), world=world, epochs=args.epochs, seed=args.seed, bucket=args.bucket)
    epoch, start_step, optimizer_step = 0, 0, 0
    if args.resume:
        package = torch.load(args.resume, map_location="cpu", weights_only=False)
        if package["metadata"]["schedule"] != schedule_key:
            raise RuntimeError(f"resume schedule differs: {package['metadata']['schedule']}")
        reportgen_fsdp.load_full_trainable_state(model, package["trainable_model"])
        reportgen_fsdp.load_full_optimizer_state(optimizer, package["optimizer"])
        scheduler.load_state_dict(package["scheduler"])
        epoch, start_step, optimizer_step = int(package["epoch"]), int(package["next_step"]), int(package["optimizer_step"])
        del package
        log(event="resumed", path=args.resume, epoch=epoch, next_step=start_step, optimizer_step=optimizer_step)

    metadata = dict(arguments={**vars(args), "visual_stream": "mrrate_mil", "mil_checkpoint": stream.checkpoint},
                    labels=labels, plan_probability=True, schedule=schedule_key,
                    writer=dict(finding_feature_dim=int(getattr(model, "finding_feature_dim", 0))))

    def save(name, next_epoch, next_step):
        t0 = time.time()
        state = reportgen_fsdp.full_trainable_state(model, main_rank)
        opt = reportgen_fsdp.full_optimizer_state(optimizer, main_rank)
        if main_rank:
            package = dict(trainable_model=state, optimizer=opt, scheduler=scheduler.state_dict(), epoch=next_epoch,
                           next_step=next_step, optimizer_step=optimizer_step, metadata=metadata)
            tmp = results / f"{name}.tmp"
            torch.save(package, tmp)
            tmp.replace(results / name)
            (results / "latest_step.json").write_text(json.dumps(dict(file=name, epoch=next_epoch, next_step=next_step,
                                                                      optimizer_step=optimizer_step, time=time.time())))
        dist.barrier(device_ids=[local])
        log(event="saved", file=name, epoch=next_epoch, next_step=next_step, optimizer_step=optimizer_step,
            seconds=round(time.time() - t0, 1))

    stopper = StopFlag(device)
    model.train()
    study_types = torch.tensor([STUDY_TYPE_BRAIN_MRI], dtype=torch.long, device=device)
    modality_empty = torch.empty((0, args.resampler_dim), dtype=torch.bfloat16, device=device)
    last_save = time.monotonic()
    steps_this_launch = 0
    stop = False
    while epoch < args.epochs:
        schedule = epoch_schedule(len(cases), sizes, world, epoch, args.seed, args.bucket)
        order = schedule[start_step:, rank]
        loader = Prefetcher(stream, cases, order, reports, depth=args.loader_depth, workers=args.loader_workers)
        window = []
        for k in range(len(order)):
            step = start_step + k
            t0 = time.time()
            case, volumes, text = loader.get(k)
            t_load = time.time() - t0
            enc = stream.encode(case, volumes)
            del volumes
            uid = case["study_uid"]
            logits = dense.get(uid, device)
            plan, features, _ = plan_inputs(plan_builder, logits, enc, geometry, labels, device, args.finding_tokens)
            t_enc = time.time() - t0 - t_load
            positives = torch.as_tensor(positives_of.get(uid, np.zeros(0, np.int64)), dtype=torch.long, device=device)
            targets = target_ids(tokenizer, text, args.max_report_tokens, device)
            consistency = plan_builder.consistency_targets(positives, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(enc.tokens, modality_empty, plan, targets, positives, logits, plan, consistency,
                            finding_features=features, dino_view_kind=enc.view_kind, dino_view_series=enc.view_series,
                            dino_study_types=study_types)
            out["loss"].backward()
            grad_norm = reportgen_fsdp.clip_grad_norm([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            steps_this_launch += 1
            replay = float((enc.logits - logits.reshape(-1)).abs().max())
            stats = torch.tensor([float(out["loss"].detach()), float(targets.numel()), float(enc.total_tokens),
                                  float(plan.numel()), replay, time.time() - t0, t_load, t_enc,
                                  torch.cuda.max_memory_allocated(device) / 2**30], device=device)
            mean = stats.clone(); dist.all_reduce(mean); mean /= world
            peak = stats.clone(); dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            window.append(mean[5].item())
            if main_rank and (optimizer_step <= 20 or optimizer_step % 10 == 0):
                log(event="step", epoch=epoch, step=step, opt_step=optimizer_step, loss=round(mean[0].item(), 4),
                    target_tokens=int(mean[1].item()), visual_tokens=int(mean[2].item()),
                    plan_tokens=int(mean[3].item()), replay_max=round(peak[4].item(), 4),
                    s_step=round(mean[5].item(), 2), s_step_max=round(peak[5].item(), 2),
                    s_load=round(mean[6].item(), 2), s_enc=round(mean[7].item(), 2),
                    grad_norm=round(float(grad_norm), 3), lr=[g["lr"] for g in optimizer.param_groups],
                    peak_gib=round(peak[8].item(), 1), s_step_avg10=round(float(np.mean(window[-10:])), 2))
            del out, enc, features, plan, targets
            done_epoch = step + 1 == len(schedule)
            stop = stopper.requested() or bool(args.stop_after_steps and steps_this_launch >= args.stop_after_steps)
            if done_epoch:
                break
            if stop:
                loader.close()
                save("latest.pt", epoch, step + 1)
                log(event="stopped", epoch=epoch, next_step=step + 1, optimizer_step=optimizer_step)
                dist.destroy_process_group()
                return 0
            if (args.save_every_steps and optimizer_step % args.save_every_steps == 0) or \
                    (time.monotonic() - last_save) / 60 >= args.save_every_minutes:
                save("latest.pt", epoch, step + 1)
                last_save = time.monotonic()
        loader.close()
        epoch += 1
        start_step = 0
        save(f"checkpoint_epoch{epoch:03d}.pt", epoch, 0)
        if main_rank:
            tmp = results / "latest.pt.lnk"
            tmp.unlink(missing_ok=True)
            os.link(results / f"checkpoint_epoch{epoch:03d}.pt", tmp)
            tmp.replace(results / "latest.pt")
        dist.barrier(device_ids=[local])
        last_save = time.monotonic()
        if stop:
            log(event="stopped", epoch=epoch, next_step=0, optimizer_step=optimizer_step)
            dist.destroy_process_group()
            return 0
    if main_rank:
        (results / "COMPLETE.json").write_text(json.dumps(dict(epochs=args.epochs, optimizer_step=optimizer_step,
                                                               time=time.time())))
    log(event="complete", optimizer_step=optimizer_step)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
