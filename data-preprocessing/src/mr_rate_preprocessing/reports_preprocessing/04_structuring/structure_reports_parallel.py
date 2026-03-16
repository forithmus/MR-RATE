# =======================================================================
# Parallel Report Structuring
# Extract brain-only sections, normalize formatting
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

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # =======================================================================
    # 0. CLI arguments and rank/world size
    # =======================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="structure_shards")
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
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"structure_rank_{RANK}.csv")

    print(f"[Rank {RANK}] Reading CSV file: {args.input_file}")
    df = pd.read_csv(args.input_file, encoding="utf-8-sig")

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
        print(f"[Rank {RANK}] All reports already structured.")
        exit(0)

    # =======================================================================
    # 3. System Prompt
    # =======================================================================
    system_prompt = """You are a medical radiology report structuring assistant. You will receive an English radiology report and must parse it into 4 structured sections. Do NOT omit or add any content — just parse and format.

IMPORTANT: Keep your thinking brief. Do NOT deliberate excessively. This is a straightforward parsing task — just identify the sections and output JSON.

CRITICAL: NEVER use "..." or ellipsis as a placeholder in any field. You MUST include the FULL actual content for every section. If a section is empty in the original report, use an empty string "". But if content exists, you MUST include ALL of it — never abbreviate with "...".

## Your task:
1. Parse the report into exactly 4 sections: clinical_information, technique, findings, impression.
2. Keep ALL content. Do NOT remove any information.
3. Normalize the formatting as specified below.

## Section extraction:
- **clinical_information**: The clinical indication/history/reason for the exam. If none exists, use "".
- **technique**: The imaging technique, sequences, contrast info. Include ALL technique info (brain, cervical, thoracic, etc.). If none exists, use "".
- **findings**: ALL observations from the report.
- **impression**: ALL conclusions from the report.

## Multi-section reports:
Many reports cover multiple exams (e.g., "BRAIN MRI, CERVICAL AND THORACIC VERTEBRA MRI"). For these:

### Findings:
- Add a section title line before each section's findings, derived from the report title.
- Example for "BRAIN AND CERVICAL VERTEBRA MRI":
  Cranial:
  [cranial findings as paragraph sentences...]

  Cervical:
  [cervical findings as paragraph sentences...]

- Example for "BRAIN MRI, CERVICAL AND THORACIC VERTEBRA MRI":
  Cranial:
  [cranial findings...]

  Cervical:
  [cervical findings...]

  Thoracic:
  [thoracic findings...]

- If the report only covers one exam (e.g., "CRANIAL MRI"), do NOT add any section title — just write the findings directly.

### Impression:
- Same logic: add section title lines for multi-section reports, no titles for single-section reports.
- Example:
  Cranial:
  — [cranial impression bullet 1]
  — [cranial impression bullet 2]

  Cervical:
  — [cervical impression bullet 1]

### How to determine section titles:
- Use the report's own title line. If it says "BRAIN AND CERVICAL VERTEBRA MRI", use "Cranial:" and "Cervical:".
- If the title says "CRANIAL, CERVICAL AND THORACIC VERTEBRA MRI", use "Cranial:", "Cervical:", "Thoracic:".
- If MR angiography is a separate section in the report, use "MR Angiography:" or similar based on what the report says.
- Use short, clean labels: "Cranial:", "Cervical:", "Thoracic:", "Lumbar:", "MR Angiography:".

## Formatting rules:

### Findings format — PARAGRAPH SENTENCES:
- Write findings as flowing paragraph text with sentences.
- Remove all bullet points (•, *, ·, -, numbered lists) from findings and convert to plain sentences.
- Each distinct observation should be a sentence. Separate paragraphs with a single newline.
- Remove excessive blank lines.
- Remove tab characters.

### Impression format — BULLET POINTS:
- Each impression item must be a bullet point using "— " (em dash followed by a space, Unicode U+2014).
- One finding per bullet point.
- If the original impression is a single sentence, still format it as "— sentence".

### General:
- Keep all content within sections exactly as-is (measurements, dates, placeholder tokens like [patient_1], [date_1], [radiologist_1]).
- Remove standalone greeting lines, title lines, and trailing signatures that are outside the 4 sections.

## Output format:
Return ONLY valid JSON, no preamble, no code fences.

If the report contains valid radiology content:
{"clinical_information": "...", "technique": "...", "findings": "...", "impression": "..."}

If the report is garbage (empty, "Thinking Process", placeholder only, not a radiology report):
{"invalid": true, "reason": "brief description"}

CRITICAL: Output ONLY the JSON. Do NOT omit any content from the report."""

    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 2000

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50000,
    )

    def parse_response(raw_text):
        """Strip thinking/fences and parse JSON from response."""
        response = raw_text.strip()
        if "</think>" in response:
            response = response[response.rfind("</think>") + len("</think>"):].strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1]
            if response.endswith("```"):
                response = response[:-3].strip()
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end]), response
        except json.JSONDecodeError:
            pass
        return None, response

    def fix_impression_bullets(text):
        """Ensure impression uses em dash (—) bullets, no leading single quotes."""
        if not text:
            return text
        lines = text.split("\n")
        fixed = []
        for line in lines:
            # Strip leading single quote
            if line.startswith("'"):
                line = line[1:]
            # Replace regular dash bullets with em dash
            if line.startswith("- "):
                line = "\u2014 " + line[2:]
            fixed.append(line)
        return "\n".join(fixed)

    def make_row(i, chunk_df, parsed, response):
        """Build output row from parsed JSON."""
        raw_report = str(chunk_df.iloc[i]["english_anonymized_report"]) if isinstance(chunk_df.iloc[i]["english_anonymized_report"], str) else ""
        base = {
            "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
            "UID": chunk_df.iloc[i]["UID"],
            "raw_report": raw_report,
        }
        if parsed is None:
            return {**base, "clinical_information": "", "technique": "", "findings": "", "impression": "",
                    "parse_status": "parse_failed", "invalid_reason": response[:200]}
        if parsed.get("invalid"):
            return {**base, "clinical_information": "", "technique": "", "findings": "", "impression": "",
                    "parse_status": "invalid", "invalid_reason": parsed.get("reason", "")}
        if "findings" in parsed or "impression" in parsed:
            return {**base,
                    "clinical_information": parsed.get("clinical_information", ""),
                    "technique": parsed.get("technique", ""),
                    "findings": parsed.get("findings", ""),
                    "impression": fix_impression_bullets(parsed.get("impression", "")),
                    "parse_status": "ok", "invalid_reason": ""}
        return {**base, "clinical_information": "", "technique": "", "findings": "", "impression": "",
                "parse_status": "parse_failed", "invalid_reason": response[:200]}

    total_processed = 0

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        prompts = []
        for _, row in chunk_df.iterrows():
            english = row["english_anonymized_report"] if isinstance(row["english_anonymized_report"], str) else "[EMPTY]"

            prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Structure this radiology report:\n\n{english}",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
        outputs = llm.generate(prompts, sampling_params)

        rows = []
        retry_indices = []
        for i, output in enumerate(outputs):
            parsed, response = parse_response(output.outputs[0].text)
            row = make_row(i, chunk_df, parsed, response)
            if row["parse_status"] == "parse_failed":
                retry_indices.append(i)
            else:
                rows.append(row)

        # Retry failures with pre-filled assistant prefix to force JSON
        if retry_indices:
            print(f"[Rank {RANK}] Retrying {len(retry_indices)} parse failures with JSON prefix...")
            retry_prompts = []
            for i in retry_indices:
                english = chunk_df.iloc[i]["english_anonymized_report"] if isinstance(chunk_df.iloc[i]["english_anonymized_report"], str) else "[EMPTY]"
                retry_prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Structure this radiology report:\n\n{english}",
                            },
                            {
                                "role": "assistant",
                                "content": "<think>\nI need to extract the brain/cranial content and output JSON.\n</think>\n\n{",
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )

            retry_params = SamplingParams(temperature=0.0, max_tokens=50000)
            retry_outputs = llm.generate(retry_prompts, retry_params)

            for j, output in enumerate(retry_outputs):
                i = retry_indices[j]
                # Prepend the "{" we used as prefix
                raw_text = "{" + output.outputs[0].text
                parsed, response = parse_response(raw_text)
                rows.append(make_row(i, chunk_df, parsed, response))

            retry_ok = sum(1 for j in range(len(retry_indices))
                          if rows[-(len(retry_indices)-j)]["parse_status"] != "parse_failed")
            print(f"[Rank {RANK}] Retry recovered {retry_ok}/{len(retry_indices)}")

        result_chunk = pd.DataFrame(rows)

        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        total_processed += len(chunk_df)

        # Quick stats
        stats = pd.DataFrame(rows)
        status_counts = stats["parse_status"].value_counts()
        print(f"[Rank {RANK}] Status: {dict(status_counts)}")
        print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

    print(f"[Rank {RANK}] Done. Structured {total_processed} reports -> {output_file}")
