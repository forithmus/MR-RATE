# =======================================================================
# NeuroVFM MRI diagnosis extraction on MR-RATE reports, via Gemma (vLLM).
#
# Faithful reproduction of the NeuroVFM "MRI diagnosis extraction prompt"
# (Supplementary Data Figure 6): an LLM-based annotation pipeline that
# converts free-text radiology reports into structured labels for the 74
# expert-defined MRI diagnoses. For EACH diagnosis the model returns a
# present/absent label ("yes"/"no") plus a one-sentence rationale.
#
# Differences from the original NeuroVFM pipeline:
#   - Runs a local open-weights Gemma model through vLLM instead of the
#     hosted GPT-4.1-mini (`--model_name`, default google/gemma-4-31B-it).
#   - Gemma has NO system role, so the instruction block + report are sent
#     as a single user turn (which is how the NeuroVFM prompt is structured
#     anyway: one instruction preamble followed by the report).
#
# Input:  MR-RATE structured report CSVs (batch{NN}_reports.csv) with
#         study_uid + findings/impression/clinical_information columns.
# Output: Per-rank JSON with {label 0/1, rationale} per diagnosis per
#         study_uid. Merge to a wide CSV with merge_labels.py.
#
# Deterministic: temperature=0 + fixed seed. SLURM-shardable: each rank
# processes its own contiguous shard (SLURM_PROCID / SLURM_NTASKS).
# Resume-safe at the rank level: skips a shard whose output already exists.
# =======================================================================

import os

# --- GPU isolation: each SLURM local task uses exactly one GPU ---
# FORCE (not setdefault): SLURM exports all allocated GPUs to every task, so a
# setdefault leaves CUDA_VISIBLE_DEVICES="0,1,2,3" and every rank stacks onto
# GPU 0 -> OOM. Pin each local task to its own device by SLURM_LOCALID.
# Launch srun with --gpu-bind=none so all GPUs are visible and this index maps.
local_id = os.environ.get("SLURM_LOCALID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = local_id

# --- HF / cache setup (mirrors the MR-RATE preprocessing scripts) ---
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./cache")
# HPC compute nodes usually have no outbound internet — read weights from the
# local HF cache only. Pre-download on the login node first (see README).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

slurm_job = os.environ.get("SLURM_JOB_ID", "local")
slurm_proc = os.environ.get("SLURM_PROCID", "0")
local_temp = f"/tmp/{slurm_job}_{slurm_proc}"
os.makedirs(local_temp, exist_ok=True)
for var, subdir in [
    ("XDG_CACHE_HOME", "xdg_cache"),
    ("XDG_CONFIG_HOME", "xdg_config"),
    ("TRITON_CACHE_DIR", "triton_cache"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
    ("VLLM_CACHE_ROOT", "vllm_cache"),
]:
    path = f"{local_temp}/{subdir}"
    os.environ.setdefault(var, path)
    os.makedirs(path, exist_ok=True)

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone


# ============================================================
# Prompt  (verbatim NeuroVFM MRI diagnosis extraction prompt)
# ============================================================

# The instruction preamble. The diagnoses list is appended where the caption
# reads "<diagnoses, refer to caption>", then the report is appended after.
NEUROVFM_PROMPT = """\
You, ChatGPT, are an expert neuroradiologist specializing in the interpretation of brain and spine MRI imaging reports. You will classify a radiology report according to the predefined list diagnoses below. This is a multilabel classification task, meaning a patient may have any number of these diagnoses (including none).

Your job is to:
1. Parse the free-text radiology report.
2. Determine the presence or absence of each diagnosis based strictly on the imaging findings and/or impression sections.
3. Output a structured JSON object where:
   - Each diagnosis is a key.
   - The value is an object with:
     - "rationale": A one-sentence explanation of the decision.
     - "label": "yes" if the diagnosis is present based on imaging findings and/or impression, "no" otherwise.

Strict Classification Guidelines:
- ONLY base the classification on the FINDINGS and IMPRESSION sections.
- Ignore the INDICATION/HISTORY section unless it directly correlates with imaging findings.
- A history of a condition does NOT mean it is present in the current imaging.
- A diagnosis should be labeled `"yes"` only if explicitly stated in the imaging findings or mentioned as a possible diagnosis.
  - Example: "There is evidence of..." -> `"yes"`
  - Example: "Findings are consistent with..." -> `"yes"`
  - Example: "Diagnoses on the differential are..." -> "yes"
- A diagnosis must be labeled `"no"` if the report explicitly negates it or states it is resolved.
  - Example: "No evidence of..." -> `"no"`
  - Example: "Previously resected..." -> `"no"`
  - Example: "Normal exam, no abnormal findings..." -> `"no"`
- If a condition is not mentioned at all in the findings/impression, classify it as `"no"` by default.
  - Example: If "brain metastasis" is not discussed in the findings, set `"brain_metastasis": {{"rationale": "No mention of brain metastasis in the findings.", "label": "no"}}`
- If the report states that there is no residual or no recurrence of a prior diagnosis, classify it as `"no"`.
  - Example: "No residual tumor seen." -> `"no"`
  - Example: "Prior infarct, but no acute ischemic stroke" -> `"acute_ischemic_stroke": "no"`

Expected JSON Output Format
{{
  "subdural_hematoma": {{
    "rationale": "No mention of subdural hematoma is present in the report.",
    "label": "no"
  }},
  "epidural_hematoma": {{
    "rationale": "The report explicitly states there is no epidural hematoma.",
    "label": "no"
  }},
  "encephalomalacia_gliosis": {{
    "rationale": "The report describes hypodense areas consistent with prior infarction, suggesting encephalomalacia.",
    "label": "yes"
  }},
  "...": {{
    "rationale": "...",
    "label": "..."
  }}
}}

Diagnoses List (Ensure a Key for Each in the JSON Output):
{diagnoses_block}

Final Notes:
- Every diagnosis must have a key in the JSON output.
- Ensure strict adherence to the format (no missing brackets, misplaced commas, colons, or inconsistent tuple usage).
- Avoid free-text responses; only structured JSON format is allowed.
"""


def build_diagnoses_block(diagnoses):
    """One line per diagnosis: `- key (guidance)` when guidance is present."""
    lines = []
    for d in diagnoses:
        g = d.get("guidance", "").strip()
        lines.append(f"- {d['key']} ({g})" if g else f"- {d['key']}")
    return "\n".join(lines)


def build_report_text(row):
    """Assemble the report from MR-RATE sections.

    The prompt classifies on FINDINGS/IMPRESSION and treats INDICATION as
    context only, so we label the sections explicitly. Falls back to the
    whole `report` column if the structured sections are empty.
    """
    findings = (row.get("findings") or "").strip()
    impression = (row.get("impression") or "").strip()
    indication = (row.get("clinical_information") or "").strip()
    if not findings and not impression:
        return (row.get("report") or "").strip()
    parts = []
    if indication:
        parts.append(f"INDICATION/HISTORY:\n{indication}")
    if findings:
        parts.append(f"FINDINGS:\n{findings}")
    if impression:
        parts.append(f"IMPRESSION:\n{impression}")
    return "\n\n".join(parts)


def build_user_message(prompt_preamble, report_text):
    return (
        f"{prompt_preamble}\n"
        f"Now classify the following radiology report. "
        f"Return ONLY the JSON object.\n\n"
        f"RADIOLOGY REPORT:\n---\n{report_text}\n---\n"
    )


# ============================================================
# Output parsing
# ============================================================

def parse_diagnosis_json(text, diagnosis_keys):
    """Parse the model's JSON into {key: (label 0/1, rationale)}.

    Robust to markdown fences and to the model returning either the nested
    {"label","rationale"} form or a bare "yes"/"no"/0/1 value. Missing keys
    default to absent (0), matching the prompt's "not mentioned -> no" rule.
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    frag = t[s:e + 1]
    try:
        data = json.loads(frag)
    except json.JSONDecodeError:
        # Gemma occasionally emits, for exactly one of the 74 keys, a rationale
        # followed by a trailing comma with no "label" field:
        #     "brain_abscess": { "rationale": "No mention of ...", },
        # Strict JSON rejects the trailing comma, which would throw away all 74
        # labels over one stray character. Strip trailing commas and retry; the
        # missing "label" then falls through to the "no" default below (which is
        # what those rationales say anyway).
        repaired = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None

    def to_label(v):
        # Nested {"label": ..., "rationale": ...}
        rationale = ""
        if isinstance(v, dict):
            rationale = str(v.get("rationale", ""))[:500]
            v = v.get("label", "no")
        # Normalize the label value
        if isinstance(v, bool):
            return (1 if v else 0), rationale
        if isinstance(v, (int, float)):
            return (1 if int(v) == 1 else 0), rationale
        s = str(v).strip().lower()
        return (1 if s in ("yes", "y", "true", "present", "1") else 0), rationale

    out = {}
    for k in diagnosis_keys:
        if k in data:
            out[k] = to_label(data[k])
        else:
            out[k] = (0, "not present in model output; defaulted to no")
    return out


# ============================================================
# Report loading + sharding
# ============================================================

def load_reports(reports_dir):
    """Load every batch{NN}_reports.csv in reports_dir (NN = 00..99)."""
    all_reports = []
    for i in range(100):
        path = os.path.join(reports_dir, f"batch{i:02d}_reports.csv")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                report_text = build_report_text(row)
                if report_text:
                    all_reports.append(
                        {"study_uid": row["study_uid"], "report": report_text}
                    )
    return all_reports


def compute_data_hash(reports):
    h = hashlib.sha256()
    for r in reports:
        h.update(r["study_uid"].encode())
        h.update(r["report"].encode())
    return h.hexdigest()[:16]


# ============================================================
# Main
# ============================================================

def main():
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser(
        description="Extract the 74 NeuroVFM MRI diagnoses from MR-RATE "
                    "reports with a local Gemma model via vLLM."
    )
    ap.add_argument("--reports_dir", required=True,
                    help="Dir with batch{NN}_reports.csv "
                         "(e.g. MR-RATE-validation/reports)")
    ap.add_argument("--diagnoses_json",
                    default=os.path.join(os.path.dirname(__file__),
                                         "neurovfm_mri_diagnoses.json"),
                    help="JSON with the 74 NeuroVFM diagnoses + guidance")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write labels_rank_{rank}.json")
    ap.add_argument("--model_name", default="google/gemma-4-31B-it",
                    help="HF model id for a Gemma instruct model "
                         "(dense google/gemma-4-31B-it, or MoE "
                         "google/gemma-4-26B-A4B for faster throughput)")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Reports per vLLM generate() call + checkpoint interval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=4096,
                    help="Max new tokens (74 rationales+labels fit in ~2.5k)")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem_frac", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: only process the first N reports (pre-shard)")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"labels_rank_{rank}.json")

    # Resume: seed this rank's own file, but skip any study_uid completed by
    # ANY rank file (global done-set). This decouples resume from the shard
    # layout, so the job can be resubmitted with a different node count after
    # a preemption and still never re-does or drops work.
    import glob as _glob
    prior_results = []
    if os.path.exists(output_path):
        try:
            prior_results = json.load(open(output_path)).get("results", [])
        except Exception as e:
            print(f"[Rank {rank}] could not read own prior output ({e})",
                  flush=True)
    done_uids = set()
    for fp in _glob.glob(os.path.join(args.output_dir, "labels_rank_*.json")):
        try:
            for r in json.load(open(fp)).get("results", []):
                done_uids.add(r["study_uid"])
        except Exception:
            pass
    print(f"[Rank {rank}] resume: {len(done_uids)} study_uids globally done; "
          f"own file has {len(prior_results)}", flush=True)

    # ---- Diagnoses ----
    with open(args.diagnoses_json) as f:
        diagnoses = json.load(f)["diagnoses"]
    diagnosis_keys = [d["key"] for d in diagnoses]
    diagnoses_block = build_diagnoses_block(diagnoses)
    prompt_preamble = NEUROVFM_PROMPT.format(diagnoses_block=diagnoses_block)
    n_dx = len(diagnosis_keys)
    print(f"[Rank {rank}] Loaded {n_dx} diagnoses", flush=True)

    # ---- Reports + shard ----
    all_reports = load_reports(args.reports_dir)
    if args.limit:
        all_reports = all_reports[:args.limit]
    total = len(all_reports)
    data_hash = compute_data_hash(all_reports)
    per_rank = math.ceil(total / world_size)
    start = rank * per_rank
    end = min(start + per_rank, total)
    shard_all = all_reports[start:end]
    shard = [it for it in shard_all if it["study_uid"] not in done_uids]
    print(f"[Rank {rank}/{world_size}] {total} reports total; shard "
          f"{start}-{end} ({len(shard_all)}); {len(shard)} remaining after "
          f"resume | hash={data_hash}", flush=True)
    if not shard:
        print(f"[Rank {rank}] nothing left to do.", flush=True)
        if not os.path.exists(output_path):
            json.dump({"metadata": {"rank": rank, "n_reports": 0}, "results": []},
                      open(output_path, "w"))
        return

    # ---- Load Gemma via vLLM ----
    print(f"[Rank {rank}] Loading {args.model_name} ...", flush=True)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_frac,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                              seed=args.seed)

    # Guard: a prompt longer than the context makes vLLM raise, which kills the
    # rank -- and srun then tears down every other rank in the job. MR-RATE has
    # a handful of pathologically long reports (max ~31k tokens vs p99.9 ~1.1k),
    # so clamp the REPORT (never the instructions) to fit. Leave room for output.
    max_prompt_tokens = max(512, args.max_model_len - args.max_tokens)
    n_truncated = 0

    def render_prompt(report_text, suffix=""):
        nonlocal n_truncated

        def _render(rt):
            # Gemma has no system role -> single user turn.
            messages = [{"role": "user",
                         "content": build_user_message(prompt_preamble, rt) + suffix}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        text = _render(report_text)
        n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tok <= max_prompt_tokens:
            return text
        overflow = n_tok - max_prompt_tokens
        rt_ids = tokenizer(report_text, add_special_tokens=False).input_ids
        keep = max(128, len(rt_ids) - overflow - 64)
        rt = (tokenizer.decode(rt_ids[:keep])
              + "\n[...report truncated to fit model context...]")
        n_truncated += 1
        print(f"[Rank {rank}] truncated an over-long report "
              f"({n_tok} -> ~{max_prompt_tokens} prompt tokens)", flush=True)
        return _render(rt)

    # ---- Process batches ----
    results = list(prior_results)   # keep already-done rows
    n_failed = 0

    def save_output():
        out = {
            "metadata": {
                "model": args.model_name,
                "prompt": "neurovfm_mri_diagnosis_extraction",
                "seed": args.seed,
                "n_diagnoses": n_dx,
                "diagnosis_keys": diagnosis_keys,
                "rank": rank,
                "world_size": world_size,
                "shard_start": start,
                "shard_end": end,
                "n_reports": len(results),
                "data_hash": data_hash,
                "unparseable": n_failed,
                "truncated_over_long": n_truncated,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
        }
        tmp = output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, output_path)   # atomic; never leaves a half-written file

    for bstart in range(0, len(shard), args.batch_size):
        batch = shard[bstart:bstart + args.batch_size]
        prompts = [render_prompt(it["report"]) for it in batch]
        outputs = llm.generate(prompts, sampling)

        # One retry for anything that didn't parse, nudging JSON-only.
        retry_idx = []
        parsed = [None] * len(batch)
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            p = parse_diagnosis_json(text, diagnosis_keys)
            if p is None:
                retry_idx.append(i)
            else:
                parsed[i] = (p, text)

        if retry_idx:
            rprompts = []
            for i in retry_idx:
                # Route through render_prompt so the length guard applies here
                # too -- otherwise a long report re-raises on the retry path.
                rprompts.append(render_prompt(
                    batch[i]["report"],
                    suffix="\n\nReturn ONLY a single valid JSON object, "
                           "nothing else."))
            routputs = llm.generate(rprompts, sampling)
            for j, out in enumerate(routputs):
                i = retry_idx[j]
                text = out.outputs[0].text
                p = parse_diagnosis_json(text, diagnosis_keys)
                # Even if still unparseable, default-all-no keeps the row.
                if p is None:
                    p = {k: (0, "unparseable model output") for k in diagnosis_keys}
                    n_failed += 1
                parsed[i] = (p, text)

        for i, it in enumerate(batch):
            p, raw = parsed[i]
            labels = {k: p[k][0] for k in diagnosis_keys}
            rationales = {k: p[k][1] for k in diagnosis_keys if p[k][1]}
            results.append({
                "study_uid": it["study_uid"],
                "labels": labels,
                "rationales": rationales,
            })

        n_pos = sum(1 for r in results[-len(batch):]
                    if any(v == 1 for v in r["labels"].values()))
        save_output()   # checkpoint after every batch
        print(f"[Rank {rank}] batch {bstart}-{bstart+len(batch)}: "
              f"positive_reports={n_pos}/{len(batch)} | "
              f"total_done={len(results)}/{len(shard_all)}", flush=True)

    # ---- Final save ----
    save_output()
    print(f"[Rank {rank}] Done. Saved {len(results)} results to {output_path} "
          f"(unparseable={n_failed})", flush=True)


if __name__ == "__main__":
    main()
