# =======================================================================
# Parallel Turkish-to-English Translation for Anonymized Reports
# Each SLURM task translates its shard on its own GPU via vLLM.
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
parser.add_argument("--output_dir", type=str, default="translated_shards")
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
# 2. Load Anonymized Reports and shard
# =======================================================================
input_file = args.input_file
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"translated_rank_{RANK}.csv")

print(f"[Rank {RANK}] Reading CSV file: {input_file}")
df = pd.read_csv(input_file, encoding="utf-8-sig")
if "Anonymized_Rapor" not in df.columns:
    raise ValueError("CSV must contain 'Anonymized_Rapor' column.")

total_reports = len(df)

# Shard: each rank takes every WORLD_SIZE-th row
df = df.iloc[RANK::WORLD_SIZE].reset_index(drop=True)
print(f"[Rank {RANK}] Total reports: {total_reports}, this shard: {len(df)}")

# Resume support: skip already-processed rows
if os.path.exists(output_file):
    done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
    done_accessions = set(done_df["AccessionNo"].astype(str))
    mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
    df = df[mask].reset_index(drop=True)
    print(f"[Rank {RANK}] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

if len(df) == 0:
    print(f"[Rank {RANK}] All reports already translated.")
    exit(0)

# =======================================================================
# 3. Prepare Prompts for Translation
# =======================================================================
system_prompt = """You are a professional medical translator specializing in Turkish to English translation.

Your task is to translate Turkish medical reports to English with the following strict requirements:

1) TRANSLATION REQUIREMENTS:
   - Translate the entire medical report accurately from Turkish to English.
   - Preserve all medical terminology with correct English equivalents.
   - Maintain the clinical accuracy and meaning of all diagnoses, findings, and measurements.
   - Keep all numerical values, dates, and units exactly as they appear.
   - Preserve the structure and formatting of the original report.

2) PRESERVED ELEMENTS:
   - All measurements and units (cm, mm, ml, etc.)
   - All dates and times
   - All numerical values and ranges
   - Medical abbreviations (translate or keep standard international ones)
   - Anatomical terms (use standard English medical terminology)

3) ANONYMIZATION TOKENS:
   - The text contains anonymization tokens like [patient_1], [radiologist_1], [date_1], [hospital_1], etc.
   - Keep ALL anonymization tokens EXACTLY as they are. Do NOT translate or modify them.

4) OUTPUT FORMAT REQUIREMENTS:
   - OUTPUT ONLY the translated English text.
   - Do NOT add any introductory phrases like "Here is the translation:" or "Translation:".
   - Do NOT add any explanatory notes or comments.
   - Do NOT include the original Turkish text.
   - Maintain paragraph structure and formatting from the original.
   - Send the translation only ONCE, do not repeat."""

print(f"[Rank {RANK}] Preparing prompts...")

reports_list = df["Anonymized_Rapor"].tolist()

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(report)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for report in reports_list
]

# =======================================================================
# 4. Batched Generation with vLLM
# =======================================================================
MAX_NEW = 30000
CHUNK_SIZE = 5000

print(f"[Rank {RANK}] Running translation ({len(df)} reports)...")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW,
)


def clean_output(text):
    """Strip thinking tags and common LLM prefixes."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    prefixes = [
        "Here is the translation:",
        "Translation:",
        "Here is the translated text:",
        "Translated text:",
        "Output:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text


# =======================================================================
# 5. Process in chunks with checkpointing
# =======================================================================
save_cols = [c for c in df.columns if c != "Translated_Rapor"] + ["Translated_Rapor"]

total_processed = 0

for chunk_start in range(0, len(df), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
    chunk_df = df.iloc[chunk_start:chunk_end].copy()
    chunk_prompts = prompts[chunk_start:chunk_end]

    print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
    outputs = llm.generate(chunk_prompts, sampling_params)

    translations = []
    for output in outputs:
        translated = clean_output(output.outputs[0].text)
        translations.append(translated)

    chunk_df["Translated_Rapor"] = translations

    result_cols = [c for c in save_cols if c in chunk_df.columns]
    result_chunk = chunk_df[result_cols]

    write_header = not os.path.exists(output_file)
    result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

    total_processed += len(chunk_df)
    print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

print(f"[Rank {RANK}] Done. Translated {total_processed} reports -> {output_file}")
