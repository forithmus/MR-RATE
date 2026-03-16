# =======================================================================
# Parallel Turkish Detection in English Reports
# Uses LLM to check if english_anonymized_report is actually Turkish
# =======================================================================

import os
import json

# --- GPU Isolation: each task uses exactly one GPU ---
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
_conda_prefix = os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
os.environ.setdefault("CUDA_HOME", _conda_prefix)
os.environ["FLASHINFER_EXTRA_LDFLAGS"] = f"-L{_conda_prefix}/lib -L{_conda_prefix}/targets/sbsa-linux/lib/stubs"

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =======================================================================
# 0. CLI arguments and rank/world size
# =======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="detect_turkish_shards")
args = parser.parse_args()

RANK = int(os.environ.get("SLURM_PROCID", "0"))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))

print(f"[Rank {RANK}/{WORLD_SIZE}] Starting on node {os.environ.get('SLURMD_NODENAME', 'local')}")

# =======================================================================
# 1. Load Model with vLLM
# =======================================================================
model_dir = "Qwen/Qwen3.5-35B-A3B-FP8"

print(f"[Rank {RANK}] Loading model with vLLM engine...")
llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    dtype="bfloat16",
)

print(f"[Rank {RANK}] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(f"[Rank {RANK}] vLLM engine and tokenizer loaded.")

# =======================================================================
# 2. Load Reports and shard
# =======================================================================
input_file = args.input_file
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"detect_rank_{RANK}.csv")

print(f"[Rank {RANK}] Reading CSV file: {input_file}")
df = pd.read_csv(input_file, encoding="utf-8-sig")

total_reports = len(df)

# Shard
df = df.iloc[RANK::WORLD_SIZE].reset_index(drop=True)
print(f"[Rank {RANK}] Total reports: {total_reports}, this shard: {len(df)}")

# Resume support
if os.path.exists(output_file):
    done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
    done_accessions = set(done_df["AccessionNo"].astype(str))
    mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
    df = df[mask].reset_index(drop=True)
    print(f"[Rank {RANK}] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

if len(df) == 0:
    print(f"[Rank {RANK}] All reports already checked.")
    exit(0)

# =======================================================================
# 3. System Prompt
# =======================================================================
system_prompt = """You are a language detection expert. You will be given a medical report that is supposed to be in English.

Your task is to determine the PRIMARY language of the report. The report may contain:
- Medical terms in Latin (e.g., "corpus callosum", "sella turcica") - these are NOT Turkish
- Imaging terms used internationally (e.g., "sagittal", "coronal", "axial", "FLAIR", "T1A", "T2A") - these are NOT Turkish
- Anatomical terms (e.g., "cerebellum", "ventricle", "hippocampus") - these are NOT Turkish

Focus on the SENTENCE STRUCTURE and GRAMMATICAL WORDS to determine the language:
- Turkish sentences end with verb suffixes like -dir, -tir, -dır, -mıştır, -mektedir, -ştir
- Turkish uses words like: ve (and), bir (a/one), ile (with), için (for), olan (that is), ancak (however), da/de/da (also)
- English sentences use: is, are, was, were, the, and, with, in, of, no, normal, observed, noted, etc.

Respond with EXACTLY one of these JSON responses:
{"language": "english"} - if the report is primarily in English
{"language": "turkish"} - if the report is primarily in Turkish
{"language": "mixed"} - if the report has significant portions in both languages

Return ONLY the JSON, no other text."""

# =======================================================================
# 4. Prepare Prompts
# =======================================================================
print(f"[Rank {RANK}] Preparing prompts...")

reports = df["english_anonymized_report"].tolist()

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Determine the language of this report:\n\n{report if isinstance(report, str) else '[EMPTY]'}",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for report in reports
]

# =======================================================================
# 5. Batched Processing
# =======================================================================
CHUNK_SIZE = 5000

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=16384,
)

total_processed = 0

for chunk_start in range(0, len(df), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
    chunk_df = df.iloc[chunk_start:chunk_end].copy()
    chunk_prompts = prompts[chunk_start:chunk_end]

    print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
    outputs = llm.generate(chunk_prompts, sampling_params)

    rows = []
    for i, output in enumerate(outputs):
        raw = output.outputs[0].text.strip()

        # Parse response
        text = raw
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()

        language = "unknown"
        # Try JSON parse first
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(text[json_start:json_end])
                language = parsed.get("language", "unknown").lower()
        except json.JSONDecodeError:
            pass

        # Fallback: search in both raw and post-think text
        if language == "unknown":
            search_text = (raw + " " + text).lower()
            if '"turkish"' in search_text or "'turkish'" in search_text or "language: turkish" in search_text:
                language = "turkish"
            elif '"english"' in search_text or "'english'" in search_text or "language: english" in search_text:
                language = "english"
            elif '"mixed"' in search_text or "'mixed'" in search_text:
                language = "mixed"

        rows.append({
            "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
            "detected_language": language,
            "raw_output": raw[:500],
        })

    result_chunk = pd.DataFrame(rows)

    write_header = not os.path.exists(output_file)
    result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

    total_processed += len(chunk_df)

    # Quick stats
    langs = pd.DataFrame(rows)["detected_language"].value_counts()
    print(f"[Rank {RANK}] Chunk stats: {dict(langs)}")
    print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

print(f"[Rank {RANK}] Done. Checked {total_processed} reports -> {output_file}")
