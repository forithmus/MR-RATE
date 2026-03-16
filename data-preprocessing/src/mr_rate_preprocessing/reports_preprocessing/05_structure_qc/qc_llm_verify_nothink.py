# =======================================================================
# LLM-based QC: verify structured content vs raw report
# Checks: (1) missing content, (2) hallucinated content, (3) wrong section
# Does NOT modify anything — only flags issues.
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
    parser.add_argument("--input_file", type=str, required=True,
                        help="structured_reports.csv with raw_report, findings, impression columns")
    parser.add_argument("--output_dir", type=str, default="qc_llm_shards")
    args = parser.parse_args()

    RANK = int(os.environ.get("SLURM_PROCID", "0"))
    WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))

    print(f"[Rank {RANK}/{WORLD_SIZE}] Starting LLM QC on node {os.environ.get('SLURMD_NODENAME', 'local')}")

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
    output_file = os.path.join(output_dir, f"qc_rank_{RANK}.csv")

    print(f"[Rank {RANK}] Reading CSV file: {args.input_file}")
    df = pd.read_csv(args.input_file, encoding="utf-8-sig")

    # Only QC reports with parse_status == "ok"
    df = df[df["parse_status"] == "ok"].reset_index(drop=True)
    total_reports = len(df)

    # Shard
    df = df.iloc[RANK::WORLD_SIZE].reset_index(drop=True)
    print(f"[Rank {RANK}] Total ok reports: {total_reports}, this shard: {len(df)}")

    # Resume support
    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
        done_accessions = set(done_df["AccessionNo"].astype(str))
        mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
        df = df[mask].reset_index(drop=True)
        print(f"[Rank {RANK}] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

    if len(df) == 0:
        print(f"[Rank {RANK}] All reports already verified.")
        exit(0)

    # =======================================================================
    # 3. QC Prompt
    # =======================================================================
    qc_prompt = """/no_think
You are a medical radiology report quality checker. You will receive:
1. The ORIGINAL raw radiology report
2. The STRUCTURED version (clinical_information, technique, findings, impression)

Your job: verify the structured version against the raw report. Check for:

A. MISSING CONTENT: Is any clinically meaningful information from the raw report absent in the structured version? Ignore formatting differences, minor word order changes, section titles (like "Cranial:", "Cervical:"), and radiologist names/signatures. Focus on clinical findings, measurements, diagnoses, and recommendations.

B. HALLUCINATED CONTENT: Does the structured version contain clinical information NOT present in the raw report? Added section titles, formatting (em dashes, bullet points), and minor rephrasing are acceptable. Invented findings, diagnoses, or measurements are NOT.

C. WRONG SECTION: Is content placed in the wrong section? (e.g., findings in impression, technique in clinical info)

## Output format:
Return ONLY valid JSON:
{"verdict": "pass" or "fail", "issues": [{"type": "missing"|"hallucinated"|"wrong_section", "detail": "brief description"}]}

- Set verdict to "pass" if no significant issues found.
- Set verdict to "fail" if any clinically meaningful content is missing, hallucinated, or misplaced.
- Minor formatting differences, section title additions, bullet formatting, and radiologist name omission are NOT issues.

CRITICAL: Output ONLY the JSON. No preamble, no code fences."""

    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 2000

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=26000,
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

    total_processed = 0
    total_pass = 0
    total_fail = 0
    total_error = 0

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        print(f"[Rank {RANK}] Chunk {chunk_start}-{chunk_end} / {len(df)} — verifying...")

        prompts = []
        for _, row in chunk_df.iterrows():
            raw = str(row["raw_report"]) if pd.notna(row["raw_report"]) else ""
            clin = str(row["clinical_information"]) if pd.notna(row["clinical_information"]) else ""
            tech = str(row["technique"]) if pd.notna(row["technique"]) else ""
            find = str(row["findings"]) if pd.notna(row["findings"]) else ""
            imp = str(row["impression"]) if pd.notna(row["impression"]) else ""

            user_msg = (
                f"## ORIGINAL RAW REPORT:\n{raw}\n\n"
                f"## STRUCTURED VERSION:\n"
                f"### Clinical Information:\n{clin}\n\n"
                f"### Technique:\n{tech}\n\n"
                f"### Findings:\n{find}\n\n"
                f"### Impression:\n{imp}"
            )

            prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": qc_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        outputs = llm.generate(prompts, sampling_params)

        rows = []
        retry_indices = []
        for i, output in enumerate(outputs):
            acc = chunk_df.iloc[i]["AccessionNo"]
            parsed, response = parse_response(output.outputs[0].text)

            if parsed is not None and "verdict" in parsed:
                verdict = parsed["verdict"]
                issues_list = parsed.get("issues", [])
                issues_str = json.dumps(issues_list) if issues_list else ""
                rows.append({
                    "AccessionNo": acc,
                    "verdict": verdict,
                    "issues": issues_str,
                    "qc_status": "ok",
                })
                if verdict == "pass":
                    total_pass += 1
                else:
                    total_fail += 1
            else:
                retry_indices.append(i)

        # Retry parse failures
        if retry_indices:
            print(f"[Rank {RANK}]   Retrying {len(retry_indices)} parse failures...")
            retry_prompts = []
            for i in retry_indices:
                raw = str(chunk_df.iloc[i]["raw_report"]) if pd.notna(chunk_df.iloc[i]["raw_report"]) else ""
                clin = str(chunk_df.iloc[i]["clinical_information"]) if pd.notna(chunk_df.iloc[i]["clinical_information"]) else ""
                tech = str(chunk_df.iloc[i]["technique"]) if pd.notna(chunk_df.iloc[i]["technique"]) else ""
                find = str(chunk_df.iloc[i]["findings"]) if pd.notna(chunk_df.iloc[i]["findings"]) else ""
                imp = str(chunk_df.iloc[i]["impression"]) if pd.notna(chunk_df.iloc[i]["impression"]) else ""

                user_msg = (
                    f"## ORIGINAL RAW REPORT:\n{raw}\n\n"
                    f"## STRUCTURED VERSION:\n"
                    f"### Clinical Information:\n{clin}\n\n"
                    f"### Technique:\n{tech}\n\n"
                    f"### Findings:\n{find}\n\n"
                    f"### Impression:\n{imp}"
                )

                retry_prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": qc_prompt},
                            {"role": "user", "content": user_msg},
                            {
                                "role": "assistant",
                                "content": '<think>\nI need to compare the raw report with the structured version.\n</think>\n\n{',
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                )

            retry_params = SamplingParams(temperature=0.0, max_tokens=26000)
            retry_outputs = llm.generate(retry_prompts, retry_params)

            for j, output in enumerate(retry_outputs):
                i = retry_indices[j]
                acc = chunk_df.iloc[i]["AccessionNo"]
                raw_text = "{" + output.outputs[0].text
                parsed, response = parse_response(raw_text)

                if parsed is not None and "verdict" in parsed:
                    verdict = parsed["verdict"]
                    issues_list = parsed.get("issues", [])
                    issues_str = json.dumps(issues_list) if issues_list else ""
                    rows.append({
                        "AccessionNo": acc,
                        "verdict": verdict,
                        "issues": issues_str,
                        "qc_status": "ok",
                    })
                    if verdict == "pass":
                        total_pass += 1
                    else:
                        total_fail += 1
                else:
                    rows.append({
                        "AccessionNo": acc,
                        "verdict": "error",
                        "issues": f"Parse failed: {response[:200]}",
                        "qc_status": "parse_error",
                    })
                    total_error += 1

        result_chunk = pd.DataFrame(rows)
        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        total_processed += len(chunk_df)
        chunk_fail = sum(1 for r in rows if r["verdict"] == "fail")
        print(f"[Rank {RANK}] Chunk done: {chunk_fail} failures — {total_processed}/{len(df)} total (pass={total_pass}, fail={total_fail}, error={total_error})")

    print(f"[Rank {RANK}] Done. {total_processed} reports verified: pass={total_pass}, fail={total_fail}, error={total_error} -> {output_file}")
