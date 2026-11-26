#!/usr/bin/env python3
"""Sentence-level diagnosis tags of the TRAIN report targets, for the DPO pair synthesis (Gemma-4-31B-it, vLLM).

Every findings text is split into sentences (``split_sentences``, the same splitter as ``dpo_pairs``) and Gemma lists,
per sentence, the diagnoses of the 74-diagnosis vocabulary it states as present, as possible, and as absent. Greedy
decoding, one process per GPU, resumable.

Output: OUT/rank<r>.jsonl, one line per study: {"study_uid", "sha256" (of the findings text), "sentences",
"tags": [{"present": [...], "possible": [...], "absent": [...]}, ...] (one per sentence, keys of the vocabulary)}.

srun -n 4 python -m mrrate_report.tag_sentences --reports "R/*.csv" --splits-csv S --labels-schema 74.json --out O
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\d-])|\n+")

SYSTEM = """You are a senior neuroradiologist. You receive the numbered sentences of the Findings section of one brain MRI
report and a vocabulary of diagnoses (key: name). For every sentence, list the vocabulary diagnoses that the sentence
states as present, as possible (suspected, cannot be excluded, differential), and as absent (explicitly denied or
described as normal for that diagnosis). Use only vocabulary keys. A sentence about anatomy, technique or a normal
structure that does not concern any vocabulary diagnosis gets empty lists. A diagnosis is only tagged if the sentence
itself supports it.

Answer with one JSON object and nothing else:
{"sentences": [{"n": <sentence number>, "present": [keys], "possible": [keys], "absent": [keys]}, ...]}
List only sentences with at least one tag."""


def split_sentences(text: str) -> list[str]:
    """Sentences (and separate lines) of a findings text; decimals such as 3.5 mm are not split."""
    return [s.strip() for s in BOUNDARY.split(text or "") if s and re.search(r"[A-Za-z]", s)]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse(raw: str, n: int, keys: set):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        data = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    tags = [{"present": [], "possible": [], "absent": []} for _ in range(n)]
    for item in data.get("sentences", []):
        try:
            k = int(item.get("n")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= k < n:
            for field in ("present", "possible", "absent"):
                tags[k][field] = sorted({str(x) for x in item.get(field, []) if str(x) in keys})
    return tags


def main(argv=None):
    from .data import read_reports
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--splits-csv", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--calibration", required=True, help="writer calibration (vocabulary keys and names)")
    ap.add_argument("--model", required=True, help="local Gemma-4-31B-it snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    calibration = json.loads(Path(a.calibration).read_text())
    keys, names = calibration["labels"], calibration["canonical_plan_labels"]
    vocabulary = "\n".join(f"{k}: {n}" for k, n in zip(keys, names))
    with open(a.splits_csv, newline="") as handle:
        wanted = {r["study_uid"] for r in csv.DictReader(handle) if r["split"] == a.split}
    reports = read_reports(a.reports)
    items = sorted((u, t) for u, t in reports.items() if u in wanted)
    if a.limit:
        items = items[: a.limit]
    items = items[rank::world]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rank{rank}.jsonl"
    done = set()
    if path.exists():
        with open(path) as handle:
            done = {json.loads(line)["study_uid"] for line in handle}
    todo = [(u, t) for u, t in items if u not in done]
    print(json.dumps(dict(rank=rank, items=len(items), done=len(done), todo=len(todo))), flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.90, dtype="bfloat16", seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=2048, seed=0)
    system = SYSTEM + "\n\nVocabulary:\n" + vocabulary
    t0, key_set = time.time(), set(keys)
    with open(path, "a") as handle:
        for s in range(0, len(todo), a.chunk):
            batch = [(u, t, split_sentences(t)) for u, t in todo[s:s + a.chunk]]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": "\n".join(f"{i + 1}. {x}" for i, x in enumerate(sents)) or "(empty)"}],
                tokenize=False, add_generation_prompt=True) for _, _, sents in batch]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for (uid, text, sents), o in zip(batch, outputs):
                tags = parse(o.outputs[0].text, len(sents), key_set)
                handle.write(json.dumps(dict(study_uid=uid, sha256=sha256_text(text), sentences=sents, tags=tags,
                                             finish=o.outputs[0].finish_reason)) + "\n")
            handle.flush()
            print(json.dumps(dict(rank=rank, done=s + len(batch), of=len(todo),
                                  per_s=round((s + len(batch)) / (time.time() - t0), 2))), flush=True)


if __name__ == "__main__":
    main()
