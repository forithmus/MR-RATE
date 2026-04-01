# =======================================================================
# Parallel Pathology Classification
# Multi-step LLM classification of brain/spine MRI reports against a
# predefined pathology list with SNOMED CT grounding.
#
# Pipeline:
#   Step 1 — CoT reasoning: model reads the report and reasons through
#            each pathology with exact quotes
#   Step 2 — JSON extraction: model converts CoT into structured 0/1 JSON
#   Step 3 — Verification: model re-checks PRESENT labels for false
#            positives caused by invalid inference
#
# Cross-validates CoT vs JSON (CoT is authoritative on disagreement).
# Retries on JSON parse failure with CoT-only fallback.
# Deterministic: seed + temperature=0 for reproducibility.
# Distributed across SLURM tasks — each rank processes its own shard.
#
# Input:  Structured report CSVs with "study_uid" and "findings" columns
# Output: Per-rank JSON with binary labels + CoT reasoning per study_uid,
#         and a merged CSV with study_uid + one column per pathology
# =======================================================================

import os
import sys

# --- GPU Isolation: each SLURM local task uses exactly one GPU ---
local_id = os.environ.get("SLURM_LOCALID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = local_id
os.environ["VLLM_USE_V1"] = "0"

# --- HF / Cache Setup ---
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"] + "/transformers"
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] + "/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"

# Cache isolation for multi-node SLURM jobs
slurm_job = os.environ.get("SLURM_JOB_ID", "local")
slurm_proc = os.environ.get("SLURM_PROCID", "0")
local_temp = f"/tmp/{slurm_job}_{slurm_proc}"
os.makedirs(local_temp, exist_ok=True)

for var, subdir in [
    ("XDG_CACHE_HOME", "xdg_cache"),
    ("XDG_CONFIG_HOME", "xdg_config"),
    ("TRITON_CACHE_DIR", "triton_cache"),
    ("TRITON_HOME", "triton_home"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
    ("PYTORCH_KERNEL_CACHE_PATH", "torch_kernels"),
]:
    path = f"{local_temp}/{subdir}"
    os.environ.setdefault(var, path)
    os.makedirs(path, exist_ok=True)

# Fix flashinfer JIT linking
_conda_prefix = os.environ.get(
    "CONDA_PREFIX", os.environ.get("CUDA_HOME", "/usr/local/cuda")
)
os.environ.setdefault("CUDA_HOME", _conda_prefix)
os.environ["FLASHINFER_EXTRA_LDFLAGS"] = (
    f"-L{_conda_prefix}/lib -L{_conda_prefix}/targets/sbsa-linux/lib/stubs"
)

print(f"[PID {os.getpid()}] Starting, CUDA_VISIBLE_DEVICES={local_id}", flush=True)

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone


# ============================================================
# Prompts
# ============================================================

STEP1_PROMPT = """\
You are an expert neuroradiologist. Read this brain/spine MRI report and determine if each pathology is PRESENT or ABSENT.

## Report:
{report}

## Pathology list ({n_pathologies} items):
{pathology_list}

## Output format:
For EACH pathology, write exactly one line:
  PathologyName → PRESENT — "quote the EXACT sentence from the report"
  PathologyName → ABSENT — not mentioned / denied / not detected

## Classification rules:
1. PRESENT = the report EXPLICITLY names this finding or a direct radiological synonym
2. ABSENT = the report explicitly denies it, says it was not detected, OR does not mention it at all
3. Explicit negation → ABSENT: "no evidence of X", "X was not detected", "without X", "X is absent"
4. Not mentioned at all → ABSENT
5. Uncertain or equivocal → ABSENT: "cannot exclude X", "possible X", "questionable X"
6. A finding described as "suggested" or "favored" by the radiologist counts as PRESENT
7. Sequelae count: "encephalomalacia from old infarct" → Encephalomalacia=PRESENT AND Cerebral infarction=PRESENT
8. Subtypes trigger parent: "lacunar infarct" → Lacunar infarct=PRESENT AND Cerebral infarction=PRESENT. "watershed infarct" → Watershed infarct=PRESENT AND Cerebral infarction=PRESENT

## CRITICAL — Do NOT infer one pathology from another:
- "ischemic-gliotic lesions" does NOT imply Cerebral infarction (chronic gliosis ≠ active/old infarct — only mark Cerebral infarction if the report says "infarct", "infarction", or "stroke")
- "ischemic-gliotic lesions" does NOT imply Lacunar infarct (same reason)
- "ischemic-gliotic lesions" does NOT imply Encephalomalacia (gliosis ≠ tissue destruction)
- "nonspecific white matter lesions" does NOT imply Demyelinating disease of central nervous system
- However, "plaque" or "plaques" in brain/spinal cord white matter DOES imply Demyelinating disease of central nervous system (in brain MRI, "plaque" = demyelinating plaque)
- Only mark Demyelinating disease of central nervous system if the report says "demyelinating", "demyelination", "MS", "multiple sclerosis", or "plaque(s)" in brain/cord context
- Otitis media does NOT imply Mastoiditis (different anatomical structures)
- Sphenoid/ethmoidal/maxillary sinusitis or mucosal thickening does NOT imply Mastoiditis (sinuses ≠ mastoid)
- Gliosis / gliotic lesion does NOT imply Encephalomalacia (gliosis = Gliosis ONLY)
- Dural thickening does NOT imply Hyperostosis of skull (dura ≠ calvarium)
- A bone lesion of unknown type does NOT imply Hemangioma of vertebral column
- "calcified falx" does NOT imply Intracranial meningioma (unless report explicitly says meningioma)
- Metastatic malignant neoplasm to brain does NOT imply Glioma (metastasis ≠ primary brain tumor)
- Glioma does NOT imply Metastatic malignant neoplasm to brain (primary ≠ metastatic)
- Mark Glioma as PRESENT if the report says "glioma", "glioblastoma", "GBM", "astrocytoma", "oligodendroglioma", "glial tumor", or "glial neoplasm"
- Rathke's pouch cyst does NOT imply Pituitary adenoma (cyst ≠ adenoma)
- Mark Pituitary adenoma as PRESENT if the report says "adenoma", "macroadenoma", or "microadenoma" in the pituitary/sellar context

## These ARE valid — do NOT mark them ABSENT:
- "gliotic foci/lesions/signal changes in white matter" = Gliosis (PRESENT)
- "partial empty sella" or "decreased pituitary height" = Empty sella syndrome (PRESENT)
- "ventricles dilated/widened/enlarged" = Ventriculomegaly (PRESENT)
- "volume loss" or "widened sulci" = Cerebral atrophy (PRESENT)
- "fluid in mastoid" or "mastoid effusion" = Mastoiditis (PRESENT — mastoid IS involved)
- "mastoiditis" explicitly stated = Mastoiditis
- "chronic mastoiditis" or "aeration loss in mastoid" or "mastoid opacification" = Chronic mastoiditis (NOT Mastoiditis unless report says "mastoiditis")
- Otitis media or sinus disease does NOT imply any mastoid pathology
- When in doubt, re-read the report carefully — both false positives and false negatives are errors

You MUST list ALL {n_pathologies} pathologies. Do NOT output JSON. Do NOT skip any."""


STEP2_PROMPT = """\
Now convert your analysis above into a JSON object.
Use EXACTLY these {n_pathologies} keys. Every value must be 0 or 1 (integer).
0 = ABSENT, 1 = PRESENT.
Output ONLY the raw JSON object — no markdown, no explanation, no extra text.

{json_template}"""


STEP3_VERIFY_PROMPT = """\
You marked these pathologies as PRESENT:
{present_list}

For each one, verify: does the report EXPLICITLY describe this pathology?
If the finding was INFERRED from a related but different condition, change it to ABSENT.

## KEEP as PRESENT — these are valid radiological descriptions:
- "gliotic foci/lesions/signal changes" → Gliosis = KEEP (gliotic IS gliosis)
- "ischemic-gliotic lesions/changes" → Gliosis = KEEP (the gliotic component IS gliosis — only infarction/encephalomalacia should be ABSENT, NOT Gliosis)
- "T2/FLAIR hyperintense gliotic lesions in white matter" → Gliosis = KEEP
- "partial empty sella" or "decreased pituitary height" or "pituitary gland is thinned/flattened" → Empty sella syndrome = KEEP
- "ventricles are dilated/widened/enlarged" → Ventriculomegaly = KEEP
- "cortical atrophy" or "volume loss" or "widened sulci" → Cerebral atrophy = KEEP
- "cerebellar atrophy" or "prominent cerebellar folia" → Cerebellar degeneration = KEEP
- "encephalomalacia" or "malacic changes" → Encephalomalacia = KEEP
- "mastoid effusion" or "fluid in mastoid" → Mastoiditis = KEEP (mastoid IS involved)
- "microhemorrhage/hemosiderin on SWI" → Silent micro-hemorrhage of brain = KEEP
- "glioma/glioblastoma/GBM/astrocytoma/oligodendroglioma/glial tumor" → Glioma = KEEP
- "macroadenoma/microadenoma/pituitary adenoma" → Pituitary adenoma = KEEP
- "compatible with X" or "consistent with X" or "favoring X" → X = KEEP (these are the radiologist's primary diagnosis)

## CHANGE TO ABSENT — these are invalid inferences:
- "nonspecific white matter lesions/foci" → Demyelinating disease of central nervous system = ABSENT (unless report says "demyelinating", "MS", "demyelination", or "plaque")
- "ischemic-gliotic lesions" → Cerebral infarction = ABSENT (chronic gliosis ≠ infarct, unless report says "infarct")
- "ischemic-gliotic lesions" → Lacunar infarct = ABSENT (same reason)
- "ischemic-gliotic lesions" → Encephalomalacia = ABSENT (gliosis ≠ malacia)
- "otitis media" or "sinus mucosal thickening" → Mastoiditis = ABSENT (different structure from mastoid)
- "dural thickening" → Hyperostosis of skull = ABSENT (dura ≠ bone)
- "nonspecific T2 hyperintensities" → Demyelinating disease of central nervous system = ABSENT (nonspecific ≠ demyelinating)
- BUT "plaque(s)" in brain/cord white matter → Demyelinating disease of central nervous system = KEEP (plaque = demyelinating plaque in brain MRI)
- "aeration loss in mastoid" / "mastoid opacification" → Chronic mastoiditis = KEEP, but Mastoiditis = ABSENT
- "mastoiditis" explicitly → Mastoiditis = KEEP
- "gliotic lesions" → Encephalomalacia = ABSENT (gliosis ≠ tissue loss)
- "bone metastasis/sclerotic lesion" → Hyperostosis of skull = ABSENT (metastasis ≠ thickening)
- "calcified falx" → Intracranial meningioma = ABSENT (unless report says "meningioma")
- "metastasis" → Glioma = ABSENT (metastasis ≠ primary brain tumor)
- "glioma/GBM/astrocytoma" → Metastatic malignant neoplasm to brain = ABSENT (primary ≠ metastatic)
- "Rathke cleft cyst" or "pars intermedia cyst" → Pituitary adenoma = ABSENT (cyst ≠ adenoma)
- "empty sella" → Pituitary adenoma = ABSENT (unless report also says adenoma)

For each pathology, write one line:
  PathologyName → KEEP (PRESENT) — [exact quote from report]
  PathologyName → CHANGE TO ABSENT — [reason: what the report actually says vs what was inferred]

Then output the corrected JSON with ALL {n_pathologies} keys:
{json_template}"""


RETRY_PROMPT = """\
Your previous output was not valid JSON. Please try again.
Output ONLY a valid JSON object with EXACTLY these {n_pathologies} keys, each with value 0 or 1:

{json_template}"""


# ============================================================
# Helpers
# ============================================================

def build_json_template(pathology_names):
    """Build a JSON template with placeholder values for the model to fill."""
    entries = ", ".join(f'"{name}": 0' for name in pathology_names)
    return "{" + entries + "}"


def parse_cot(raw_text, pathology_names):
    """Parse CoT output to extract PRESENT/ABSENT labels with robust matching."""
    labels = {}
    lines = raw_text.split("\n")

    for name in pathology_names:
        name_lower = name.lower()
        found = False

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower or name_lower not in line_lower:
                continue

            present_markers = [
                "→ present", "→present", "-> present", "->present",
                "– present", "— present", ": present",
            ]
            absent_markers = [
                "→ absent", "→absent", "-> absent", "->absent",
                "– absent", "— absent", ": absent",
                "not mentioned", "not detected", "denied", "no evidence",
            ]

            for marker in present_markers:
                if marker in line_lower:
                    labels[name] = 1
                    found = True
                    break
            if found:
                break

            for marker in absent_markers:
                if marker in line_lower:
                    labels[name] = 0
                    found = True
                    break
            if found:
                break

        if name not in labels:
            labels[name] = 0  # default absent if not found in CoT

    return labels


def parse_json_output(raw_text, pathology_names):
    """Parse JSON from step 2 output. Returns dict or None on failure."""
    # Strip thinking tags and markdown fences
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    result = _try_parse(text, pathology_names)
    if result is not None:
        return result

    # Try extracting largest JSON object
    best = None
    best_len = 0
    for m in re.finditer(r"\{", text):
        remainder = text[m.start():]
        depth = 0
        for ci, ch in enumerate(remainder):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = _try_parse(remainder[: ci + 1], pathology_names)
                    if candidate is not None and len(candidate) > best_len:
                        best = candidate
                        best_len = len(candidate)
                    break
    return best


def _try_parse(text, pathology_names):
    """Try to parse text as JSON and normalize to 0/1 labels."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    normalized = {}
    for k, v in obj.items():
        if isinstance(v, bool):
            normalized[k] = 1 if v else 0
        elif isinstance(v, (int, float)):
            normalized[k] = 1 if v else 0
        elif isinstance(v, str):
            normalized[k] = 1 if v.strip().lower() in ("1", "true", "present", "yes") else 0
        else:
            normalized[k] = 0
    return normalized


def cross_validate(json_labels, cot_labels, pathology_names):
    """Compare JSON and CoT labels. CoT is authoritative when they disagree.

    Returns (final_labels, list_of_disagreements).
    """
    final = {}
    disagreements = []

    for name in pathology_names:
        j = json_labels.get(name, 0)
        c = cot_labels.get(name, 0)
        if j == c:
            final[name] = j
        else:
            # CoT has explicit reasoning -> trust it over bare JSON
            final[name] = c
            disagreements.append(name)

    return final, disagreements


def load_reports(reports_dir):
    """Load all batch report CSVs in deterministic order.

    Reads the "findings" column from each batch CSV. Reports without
    findings text are skipped.
    """
    all_reports = []
    for i in range(28):
        path = os.path.join(reports_dir, f"batch{i:02d}_reports.csv")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                report = row.get("findings", "").strip()
                if report:
                    all_reports.append(
                        {"study_uid": row["study_uid"], "report": report}
                    )
    return all_reports


def compute_data_hash(reports):
    """Hash the input data for reproducibility verification."""
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

    parser = argparse.ArgumentParser(
        description="Classify MR reports against pathology list using LLM"
    )
    parser.add_argument("--reports_dir", type=str, required=True,
                        help="Directory containing batch{NN}_reports.csv files")
    parser.add_argument("--pathologies_json", type=str, required=True,
                        help="JSON file with pathology definitions")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-rank JSON and merged CSV")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-35B-A3B",
                        help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Reports per vLLM batch")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Max retries for failed JSON parse")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load pathologies ----
    with open(args.pathologies_json) as f:
        pathologies = json.load(f)["pathologies"]
    pathology_names = list(pathologies.keys())
    pathology_list_str = "\n".join(f"- {name}" for name in pathology_names)
    json_template = build_json_template(pathology_names)
    n_path = len(pathology_names)
    print(f"Loaded {n_path} pathologies", flush=True)

    # ---- Load and shard reports ----
    all_reports = load_reports(args.reports_dir)
    total = len(all_reports)
    data_hash = compute_data_hash(all_reports)
    per_rank = math.ceil(total / world_size)
    start = rank * per_rank
    end = min(start + per_rank, total)
    shard = all_reports[start:end]
    print(f"[Rank {rank}/{world_size}] Processing {start}-{end} "
          f"({len(shard)} reports) | data_hash={data_hash}", flush=True)

    # ---- Load LLM ----
    print(f"[Rank {rank}] Loading LLM {args.model_name}...", flush=True)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=40960,
        trust_remote_code=True,
        dtype="auto",
    )
    print(f"[Rank {rank}] LLM loaded", flush=True)
    tokenizer = llm.get_tokenizer()

    sampling_cot = SamplingParams(
        temperature=0.0, max_tokens=8192, seed=args.seed,
    )
    sampling_json = SamplingParams(
        temperature=0.0, max_tokens=4096, seed=args.seed,
    )

    # ---- Process batches ----
    all_results = []
    total_json_ok = 0
    total_cot_fallback = 0
    total_retries = 0
    total_disagreements = 0
    total_verified = 0
    total_flipped = 0

    for batch_start in range(0, len(shard), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(shard))
        batch = shard[batch_start:batch_end]

        # ==== Step 1: CoT reasoning ====
        step1_prompts = []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": STEP1_PROMPT.format(
                        report=item["report"],
                        pathology_list=pathology_list_str,
                        n_pathologies=n_path,
                    ),
                }
            ]
            step1_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        step1_outputs = llm.generate(step1_prompts, sampling_cot)

        # ==== Step 2: JSON extraction ====
        step2_prompts = []
        step1_texts = []
        for i, output in enumerate(step1_outputs):
            cot_text = output.outputs[0].text
            step1_texts.append(cot_text)

            messages = [
                {
                    "role": "user",
                    "content": STEP1_PROMPT.format(
                        report=batch[i]["report"],
                        pathology_list=pathology_list_str,
                        n_pathologies=n_path,
                    ),
                },
                {"role": "assistant", "content": cot_text},
                {
                    "role": "user",
                    "content": STEP2_PROMPT.format(
                        n_pathologies=n_path,
                        json_template=json_template,
                    ),
                },
            ]
            step2_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        step2_outputs = llm.generate(step2_prompts, sampling_json)

        # ==== Parse step 2 JSON ====
        n_json_ok = 0
        n_cot_fallback = 0
        n_retries = 0
        n_disagreements = 0
        n_verified = 0
        n_flipped = 0

        # Collect indices that need retry
        retry_indices = []
        parsed_results = [None] * len(batch)
        step2_labels = [None] * len(batch)

        for i, output in enumerate(step2_outputs):
            json_text = output.outputs[0].text
            cot_text = step1_texts[i]

            json_labels = parse_json_output(json_text, pathology_names)
            cot_labels = parse_cot(cot_text, pathology_names)

            if json_labels is not None:
                missing = [n for n in pathology_names if n not in json_labels]
                if missing:
                    for m in missing:
                        json_labels[m] = cot_labels.get(m, 0)

                final, disagreements = cross_validate(
                    json_labels, cot_labels, pathology_names
                )
                n_json_ok += 1
                n_disagreements += len(disagreements)
                step2_labels[i] = final
            else:
                retry_indices.append(i)

        # ==== Retry failed JSON parses ====
        for attempt in range(args.max_retries):
            if not retry_indices:
                break
            n_retries += len(retry_indices)

            retry_prompts = []
            for i in retry_indices:
                messages = [
                    {
                        "role": "user",
                        "content": STEP1_PROMPT.format(
                            report=batch[i]["report"],
                            pathology_list=pathology_list_str,
                            n_pathologies=n_path,
                        ),
                    },
                    {"role": "assistant", "content": step1_texts[i]},
                    {
                        "role": "user",
                        "content": RETRY_PROMPT.format(
                            n_pathologies=n_path,
                            json_template=json_template,
                        ),
                    },
                ]
                retry_prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            retry_outputs = llm.generate(retry_prompts, sampling_json)

            still_failed = []
            for j, output in enumerate(retry_outputs):
                i = retry_indices[j]
                json_text = output.outputs[0].text
                json_labels = parse_json_output(json_text, pathology_names)

                if json_labels is not None:
                    cot_labels = parse_cot(step1_texts[i], pathology_names)
                    for m in pathology_names:
                        if m not in json_labels:
                            json_labels[m] = cot_labels.get(m, 0)
                    final, disagreements = cross_validate(
                        json_labels, cot_labels, pathology_names
                    )
                    n_json_ok += 1
                    n_disagreements += len(disagreements)
                    step2_labels[i] = final
                else:
                    still_failed.append(i)

            retry_indices = still_failed

        # CoT-only fallback for anything still failed
        for i in retry_indices:
            cot_labels = parse_cot(step1_texts[i], pathology_names)
            n_cot_fallback += 1
            step2_labels[i] = cot_labels

        # ==== Step 3: Verify PRESENT labels ====
        # Only verify entries that have at least one PRESENT label
        verify_indices = []
        for i in range(len(batch)):
            present = [n for n in pathology_names if step2_labels[i].get(n) == 1]
            if present:
                verify_indices.append(i)

        if verify_indices:
            step3_prompts = []
            for i in verify_indices:
                present = [n for n in pathology_names if step2_labels[i].get(n) == 1]
                present_list = "\n".join(f"- {p}" for p in present)
                messages = [
                    {
                        "role": "user",
                        "content": STEP1_PROMPT.format(
                            report=batch[i]["report"],
                            pathology_list=pathology_list_str,
                            n_pathologies=n_path,
                        ),
                    },
                    {"role": "assistant", "content": step1_texts[i]},
                    {
                        "role": "user",
                        "content": STEP3_VERIFY_PROMPT.format(
                            present_list=present_list,
                            n_pathologies=n_path,
                            json_template=json_template,
                        ),
                    },
                ]
                step3_prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            step3_outputs = llm.generate(step3_prompts, sampling_json)

            for j, output in enumerate(step3_outputs):
                i = verify_indices[j]
                verify_text = output.outputs[0].text
                verified_labels = parse_json_output(verify_text, pathology_names)

                if verified_labels is not None:
                    # Only allow flips from 1→0 (removing false positives)
                    for name in pathology_names:
                        old_val = step2_labels[i].get(name, 0)
                        new_val = verified_labels.get(name, 0)
                        if old_val == 1 and new_val == 0:
                            step2_labels[i][name] = 0
                            n_flipped += 1
                    n_verified += 1

        # ==== Build final results ====
        for i in range(len(batch)):
            source = "cot_only" if i in retry_indices else "json+cot"
            if i in verify_indices:
                source += "+verified"
            parsed_results[i] = {
                "study_uid": batch[i]["study_uid"],
                "labels": step2_labels[i],
                "cot": step1_texts[i],
                "source": source,
            }

        all_results.extend(parsed_results)
        total_json_ok += n_json_ok
        total_cot_fallback += n_cot_fallback
        total_retries += n_retries
        total_disagreements += n_disagreements
        total_verified += n_verified
        total_flipped += n_flipped

        n_positive = sum(
            1 for r in parsed_results if any(v == 1 for v in r["labels"].values())
        )
        print(
            f"[Rank {rank}] Batch {batch_start}-{batch_end}: "
            f"json_ok={n_json_ok} cot_fallback={n_cot_fallback} "
            f"verified={n_verified} flipped={n_flipped} "
            f"positive={n_positive}",
            flush=True,
        )

    # ---- Save per-rank JSON ----
    output_data = {
        "metadata": {
            "model": args.model_name,
            "seed": args.seed,
            "pathologies_json": os.path.basename(args.pathologies_json),
            "n_pathologies": n_path,
            "rank": rank,
            "world_size": world_size,
            "shard_start": start,
            "shard_end": end,
            "n_reports": len(all_results),
            "data_hash": data_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "json_ok": total_json_ok,
                "cot_fallback": total_cot_fallback,
                "retries": total_retries,
                "disagreements": total_disagreements,
                "verified": total_verified,
                "flipped_to_absent": total_flipped,
            },
        },
        "results": all_results,
    }

    output_path = os.path.join(args.output_dir, f"labels_rank_{rank}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(
        f"[Rank {rank}] Done! Saved {len(all_results)} results to {output_path} | "
        f"json_ok={total_json_ok} cot_fallback={total_cot_fallback} "
        f"verified={total_verified} flipped={total_flipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
