#!/usr/bin/env python3
"""Greedy Findings generation with an MR writer checkpoint (torchrun; every rank holds the whole Gemma, no FSDP).

Visual prefix from all MIL tokens of the study, frozen-Platt label tokens, plan text with evidence descriptions and
evidence tokens, greedy decoding (1 beam, up to 1,024 new tokens) with the Gemma-4 stop ids. Every model setting
comes from the checkpoint metadata; paths can be overridden. Output: one CSV (rank 0 merges) with study_uid,
findings_gt, findings_pred and the planned findings, readable by ``report-generation-training``'s evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

PATH_KEYS = ("bundle", "data_folder", "splits_csv", "reports", "calibration", "dense")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--llm_path", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--out", required=True, help="output CSV")
    for key in PATH_KEYS:
        ap.add_argument(f"--{key}", default=None, help=f"override the {key} recorded in the checkpoint")
    a = ap.parse_args(argv)
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(hours=4))

    package = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = package["metadata"]
    targs = argparse.Namespace(**meta["arguments"])
    for key in PATH_KEYS:
        if getattr(a, key) is not None:
            setattr(targs, key, getattr(a, key))
    from .mil import MRVisualStream, add_bundle_paths
    add_bundle_paths(targs.bundle)
    from fora.reportgen.cxr.model import load_trainable, stop_ids
    from fora.reportgen.writer.model import generation_inputs

    from .data import DenseLogits, read_cases, read_reports
    from .model import STUDY_TYPE_BRAIN_MRI, build_report_model, load_llm, plan_inputs
    from .plan import MRPlanBuilder
    from .train_sft import evidence_geometry
    calibration = json.loads(Path(targs.calibration).read_text())
    labels = list(calibration["labels"])
    dense = DenseLogits(targs.dense, labels, calibration)
    stream = MRVisualStream(targs.bundle, device)
    geometry = evidence_geometry(stream)
    llm, tokenizer, llm_dim = load_llm(a.llm_path, device, targs.lora_r, targs.lora_alpha)
    plan_builder = MRPlanBuilder(targs.calibration, labels, tokenizer)
    model = build_report_model(llm, tokenizer, llm_dim, labels, plan_builder, device, targs)
    load_trainable(model, package["trainable_model"], device)
    del package
    model.eval()
    model.llm.config.use_cache = True
    stops = stop_ids(tokenizer, a.llm_path)
    study_types = torch.tensor([STUDY_TYPE_BRAIN_MRI], dtype=torch.long, device=device)
    reports = read_reports(targs.reports)
    cases = [c for c in read_cases(targs.data_folder, targs.splits_csv, (a.split,), "atlas_space") if c["study_uid"] in dense]
    if a.limit:
        cases = cases[: a.limit]
    records = []
    for case in cases[rank::world]:
        t0 = time.time()
        uid = case["study_uid"]
        enc = stream.encode(case, stream.prepare(case))
        logits = dense.get(uid, device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prefix = model.visual_prefix_from_bag(enc, study_types)
            plan, features, selection = plan_inputs(plan_builder, logits, enc, geometry, labels, device,
                                                    targs.finding_tokens)
            embeddings, attention, token_types = generation_inputs(model, prefix, logits, plan, features)
            embedding = model.llm.get_input_embeddings()
            ids = model.llm.generate(inputs_embeds=embeddings.to(embedding.weight.dtype), attention_mask=attention,
                                     token_type_ids=token_types, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     num_beams=1, use_cache=True, eos_token_id=stops, pad_token_id=stops[0],
                                     logits_to_keep=1)
        text = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        text = re.sub(r"^\W*findings\W*:?\W*", "", text, flags=re.I).strip() if re.match(r"^\W*findings\b", text, re.I) else text
        records.append(dict(study_uid=uid, findings_gt=reports.get(uid, ""), findings_pred=text,
                            confident="|".join(labels[i] for i in selection.confident),
                            possible="|".join(labels[i] for i in selection.possible),
                            new_tokens=int(ids.shape[-1]), visual_tokens=enc.total_tokens,
                            seconds=round(time.time() - t0, 1)))
        print(json.dumps(dict(event="generated", rank=rank, uid=uid, seconds=records[-1]["seconds"],
                              new_tokens=records[-1]["new_tokens"])), flush=True)
        del enc
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    if rank == 0:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        rows = [r for part in gathered for r in part]
        with open(tmp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["study_uid"])
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out)
        print(json.dumps(dict(event="done", records=len(rows), out=str(out))), flush=True)
    dist.barrier(device_ids=[local])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
