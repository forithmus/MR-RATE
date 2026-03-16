# =======================================================================
# Parallel Translation Quality Check
# Compares Turkish and English reports sentence by sentence
# Saves thinking tokens separately for review
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

# Fix flashinfer JIT linking: conda env has libs in lib/ not lib64/
_conda_prefix = os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
os.environ.setdefault("CUDA_HOME", _conda_prefix)
os.environ["FLASHINFER_EXTRA_LDFLAGS"] = f"-L{_conda_prefix}/lib -L{_conda_prefix}/targets/sbsa-linux/lib/stubs"

import argparse
import pandas as pd

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # =======================================================================
    # 0. CLI arguments and rank/world size
    # =======================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="quality_check_shards")
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

    output_file = os.path.join(output_dir, f"qc_rank_{RANK}.csv")

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
    system_prompt = """You are a bilingual (Turkish-English) medical radiology translation quality reviewer.

You will receive a Turkish radiology report and its English translation. Your job is to find REAL translation errors that change clinical meaning.

IMPORTANT: You must be VERY conservative. Only flag errors you are 100% certain about. When in doubt, PASS.

CRITICAL RULE: Before reporting ANY issue, you MUST re-read the English translation to confirm the error actually exists. Do NOT report an error from memory — go back and verify the exact English text contains the problem you think it does. Many errors you think you see will not actually be there when you look again.

## What counts as a REAL error (FAIL):
- WRONG LATERALITY: "sağ" (right) translated as "left", or vice versa, or laterality added/removed
- WRONG ANATOMY: one anatomical structure translated as a completely different one (e.g., "amygdala" → "insular cortex", "orbita içi" → "intraosseous")
- WRONG NUMBERS: measurements or counts changed (e.g., "6 mm" → "8 mm", "three lesions" → "two lesions")
- MISSING FINDINGS: an entire clinical finding or diagnosis is absent from the English (not just a single word)
- OPPOSITE MEANING: "removed" translated as "placed", "aeration" as "opacification", "or" as "and" when it changes differential diagnosis
- WRONG CLINICAL TERM: "lung cancer" as "acute ca", "vasogenic edema" as "vascular edema", "canalicular" as "intracranial"

## What is NOT an error (PASS these):
- Spelling variants: "disk"/"disc", "millimeter"/"milimeter", "perimesencephalic"/"permesencephalic"
- Synonym choices: "observed"/"noted"/"seen", "medulla oblongata"/"medulla"/"bulbus", "globe"/"bulbus oculi"
- Word order differences that preserve meaning
- Missing greeting/closing phrases ("Saygılarımla", "Sayın Meslektaşım")
- Section header format differences
- Minor grammatical differences (singular/plural when context is clear, adjective/noun forms)
- Technique section details (contrast agent names, doses, sequence parameters)
- Placeholder tokens like [patient_1], [date_1]
- Brainstem structure list order differences
- "3 planar" vs "3-plane", "SWI" vs "hemorrhage-sensitive sequence"

## Response format:
For each issue you report, you MUST quote the exact Turkish text AND the exact English text that contains the error. If you cannot find the error in the actual English text, do NOT report it.

{"verdict": "PASS", "issues": []}
or
{"verdict": "FAIL", "issues": ["EXACT Turkish: '...' → EXACT English: '...' — [description of error]"]}

Return ONLY the JSON."""

    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 2000

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=24576,
    )

    total_processed = 0

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        prompts = []
        for _, row in chunk_df.iterrows():
            turkish = row["turkish_anonymized_report"] if isinstance(row["turkish_anonymized_report"], str) else "[EMPTY]"
            english = row["english_anonymized_report"] if isinstance(row["english_anonymized_report"], str) else "[EMPTY]"

            prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"TURKISH ORIGINAL:\n{turkish}\n\n---\n\nENGLISH TRANSLATION:\n{english}",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
        outputs = llm.generate(prompts, sampling_params)

        rows = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()

            # Split thinking and response
            thinking = ""
            response = raw
            if "</think>" in raw:
                think_end = raw.rfind("</think>")
                thinking = raw[:think_end].strip()
                # Remove <think> tag if present
                if thinking.startswith("<think>"):
                    thinking = thinking[len("<think>"):].strip()
                response = raw[think_end + len("</think>"):].strip()

            # Parse verdict
            verdict = "unknown"
            issues = ""
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json.loads(response[json_start:json_end])
                    verdict = parsed.get("verdict", "unknown").upper()
                    issues_list = parsed.get("issues", [])
                    if isinstance(issues_list, list):
                        issues = " | ".join(str(x) for x in issues_list)
                    else:
                        issues = str(issues_list)
            except json.JSONDecodeError:
                pass

            # Fallback parsing
            if verdict == "unknown":
                resp_lower = (raw + " " + response).lower()
                if '"pass"' in resp_lower or "'pass'" in resp_lower:
                    verdict = "PASS"
                elif '"fail"' in resp_lower or "'fail'" in resp_lower:
                    verdict = "FAIL"

            rows.append({
                "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
                "UID": chunk_df.iloc[i]["UID"],
                "verdict": verdict,
                "issues": issues,
                "thinking": thinking,
                "response": response,
            })

        result_chunk = pd.DataFrame(rows)

        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        total_processed += len(chunk_df)

        # Quick stats
        verdicts = pd.DataFrame(rows)["verdict"].value_counts()
        print(f"[Rank {RANK}] Chunk stats: {dict(verdicts)}")
        print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

    print(f"[Rank {RANK}] Done. Checked {total_processed} reports -> {output_file}")
