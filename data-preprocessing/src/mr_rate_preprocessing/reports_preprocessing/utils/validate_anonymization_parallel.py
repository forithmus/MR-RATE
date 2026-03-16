# =======================================================================
# Parallel Anonymization Validation Script for Turkish Reports
# Each SLURM task validates its shard on its own GPU via vLLM.
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
parser.add_argument("--output_dir", type=str, default="validation_shards")
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

output_file = os.path.join(output_dir, f"validation_rank_{RANK}.csv")

print(f"[Rank {RANK}] Reading CSV file: {input_file}")
df = pd.read_csv(input_file, encoding="utf-8-sig")
if "Anonymized_Rapor" not in df.columns:
    raise ValueError("CSV must contain 'Anonymized_Rapor' column.")

total_reports = len(df)

# Shard: each rank takes every WORLD_SIZE-th row
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
    print(f"[Rank {RANK}] All reports already validated.")
    exit(0)

# =======================================================================
# 3. Prepare Validation Prompts
# =======================================================================
validation_system_prompt = """
Sen bir anonimlestirme kalite kontrol uzmanisin. Sana anonimlestirilmis bir Turkce tibbi rapor verilecek.
Gorevin, raporda hala kisisel veya kurumsal bilgi (PII) kalip kalmadigini kontrol etmektir.

Asagidaki bilgi turlerinin tokenlarla degistirilmis olmasi gerekir:
1. Kisi isimleri (hasta, doktor, hemsire, teknisyen vb.) -> [patient_N], [radiologist_N], [referring_dr_N], [staff_N], [person_N]
2. Tarihler (gun/ay/yil, ay/yil, sadece yil) -> [date_N]
3. Medipol grubu hastane isimleri -> [hospital_N]
4. Harici hastane isimleri -> [hospital_eN]
5. Accession numaralari -> [accession_N]

Korunmasi gereken (tokenlenmamasi gereken) bilgiler:
- Yas, cinsiyet
- Tibbi icerik (tani, tedavi, bulgular, olcumler, ilaclar, prosedurler)
- Sehir / ilce / mahalle isimleri
- Cihaz ve marka isimleri
- Protokol bilgileri
- Ilac isimleri ve dozlar

Raporun dogru anonimlestirilip anonimlestirilmedigini degerlendir.

CIKTI FORMATI (tam olarak bu JSON formatinda yanit ver, baska hicbir sey yazma):
{
  "result": "PASS" veya "FAIL",
  "leaked_items": [
    {"type": "NAME|DATE|HOSPITAL|ACCESSION|OTHER", "value": "sizan deger", "context": "degerin gectigi kisa baglam"}
  ],
  "notes": "varsa ek aciklama, yoksa bos string"
}

Eger rapor tamamen dogru anonimlestirilmisse:
{"result": "PASS", "leaked_items": [], "notes": ""}

Eger herhangi bir PII sizintisi varsa "FAIL" de ve sizan ogeleri listele.
SADECE JSON dondur, baska aciklama veya metin YAZMA.
"""

print(f"[Rank {RANK}] Preparing validation prompts...")

anonymized_reports = df["Anonymized_Rapor"].tolist()

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": validation_system_prompt},
            {
                "role": "user",
                "content": (
                    "Asagidaki anonimlestirilmis raporu kontrol et:\n\n"
                    f"{report if isinstance(report, str) else '[BOS RAPOR]'}"
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for report in anonymized_reports
]

# =======================================================================
# 4. Batched Validation with vLLM
# =======================================================================
MAX_NEW = 2048
CHUNK_SIZE = 5000

print(f"[Rank {RANK}] Running validation ({len(df)} reports)...")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW,
)


def parse_validation_output(raw_text):
    text = raw_text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            parsed = json.loads(json_str)
            result = parsed.get("result", "UNKNOWN").upper()
            leaked = parsed.get("leaked_items", [])
            notes = parsed.get("notes", "")
            return result, leaked, notes, json_str
    except json.JSONDecodeError:
        pass

    upper = text.upper()
    if "PASS" in upper and "FAIL" not in upper:
        return "PASS", [], "", text
    elif "FAIL" in upper:
        return "FAIL", [], "JSON parse failed - manual review needed", text
    else:
        return "UNKNOWN", [], "Could not parse LLM output - manual review needed", text


# =======================================================================
# 5. Process in chunks with checkpointing
# =======================================================================
total_processed = 0

for chunk_start in range(0, len(df), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
    chunk_df = df.iloc[chunk_start:chunk_end].copy()
    chunk_prompts = prompts[chunk_start:chunk_end]

    print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
    outputs = llm.generate(chunk_prompts, sampling_params)

    rows = []
    for i, output in enumerate(outputs):
        raw = output.outputs[0].text
        result, leaked, notes, parsed_json = parse_validation_output(raw)
        rows.append({
            "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
            "Anonymized_Rapor": chunk_df.iloc[i]["Anonymized_Rapor"],
            "final_result": result,
            "leaked_items": json.dumps(leaked, ensure_ascii=False) if leaked else "",
            "notes": notes,
            "raw_llm_output": raw,
        })

    result_chunk = pd.DataFrame(rows)

    write_header = not os.path.exists(output_file)
    result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

    total_processed += len(chunk_df)
    print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

print(f"[Rank {RANK}] Done. Validated {total_processed} reports -> {output_file}")
