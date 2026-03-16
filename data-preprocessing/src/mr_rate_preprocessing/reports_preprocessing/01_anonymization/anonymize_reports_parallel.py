# =======================================================================
# Parallel Qwen Anonymization Script for Turkish Reports
# Each SLURM task processes its shard of the data on its own GPU.
# =======================================================================

import os
import json

# --- GPU Isolation: each task uses exactly one GPU ---
# Must be set BEFORE any CUDA/torch/vllm import
local_id = os.environ.get("SLURM_LOCALID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = local_id
os.environ["VLLM_USE_V1"] = "0"

# --- HF / Cache Setup (must be set before importing transformers/vllm) ---
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
parser.add_argument("--input_file", type=str, default="merged_filtered_with_uid.csv")
parser.add_argument("--output_dir", type=str, default="anonymized_shards")
args = parser.parse_args()

RANK = int(os.environ.get("SLURM_PROCID", "0"))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))

# Each task uses its local GPU (CUDA_VISIBLE_DEVICES is set by srun)
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
# 2. Load Turkish Reports and shard
# =======================================================================
input_file = args.input_file
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"anonymized_rank_{RANK}.csv")
mapping_file = os.path.join(output_dir, f"mapping_rank_{RANK}.csv")

print(f"[Rank {RANK}] Reading CSV file: {input_file}")
df = pd.read_csv(input_file, encoding="utf-8-sig")
if not {"AccessionNo", "RaporText"}.issubset(df.columns):
    raise ValueError("CSV must contain 'AccessionNo' and 'RaporText' columns.")

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
    print(f"[Rank {RANK}] All reports already processed.")
    exit(0)

# =======================================================================
# 3. Prepare Prompts for Anonymization
# =======================================================================
system_prompt = """
Sen bir Turkce tibbi rapor anonimlesirme aracisin. Gorulen kisisel ve kurumsal bilgileri belirli tokenlarla degistiriyorsun.
Yorumlama, aciklama, ozetleme, ek cumle ekleme veya format degisikligi YAPMA.
Sadece asagidaki kurallara gore metin icindeki ilgili bilgileri tokenlarla degistir.

# KURALLAR:

## 1) KISI ISIMLERI -> Token ile degistir
   - Hasta isimleri (tek isim veya ad+soyad): [patient_1], [patient_2], ... [patient_n]
     Ayni hasta ismi tekrar gecerse ayni token kullanilmali.
   - Radyolog / raporu yazan doktor isimleri: [radiologist_1], [radiologist_2], ... [radiologist_n]
   - Sevk eden / isteyen doktor isimleri: [referring_dr_1], [referring_dr_2], ... [referring_dr_n]
   - Diger saglik personeli (hemsire, teknisyen vb.): [staff_1], [staff_2], ... [staff_n]
   - Isimle birlikte gelen tum unvanlar da token kapsamina alinir:
     "Dr.", "Prof.", "Doc.", "Uzm.", "Op.", "MD", "Radyolog", "Hemsire", "Teknisyen" vb.
     Unvan + isim birlikte tek bir token ile degistirilir.
   - Eger bir ismin rolu (hasta mi, doktor mu) baglamdan anlasilamiyorsa [person_1], [person_2] kullan.

## 2) TARIHLER -> Token ile degistir
   - Tum tarihler (gun/ay/yil, ay/yil, sadece yil dahil): [date_1], [date_2], ... [date_n]
   - Ayni tarih tekrar gecerse ayni token kullanilmali.
   - Ornekler: "01.03.2024" -> [date_1], "Mart 2024" -> [date_2], "2023" -> [date_3]

## 3) HASTANE ISIMLERI -> Token ile degistir
   Asagidaki Medipol hastaneleri dahili (internal) hastanelerdir. Bunlari [hospital_1], [hospital_2], ... [hospital_n] olarak tokenla (her farkli dahili hastaneye farkli numara ver, ayni hastane tekrarlarsa ayni token kullan):
   - Medipol Mega Universitesi Hastanesi
   - Medipol Kosuyolu Hastanesi
   - Medipol Bahcelievler Universite Hastanesi
   - MEDIPOL Acibadem Bolge Hastanesi
   - Medipol Esenler Universite Hastanesi
   - Medipol Camlica Universite Hastanesi
   - Medipol Pendik Universite Hastanesi
   - Medipol Vatan Universite Hastanesi
   - Medipol Sefakoy Universite Hastanesi
   - Medipol Unkapani Universite Dis Hastanesi
   - Medipol Ankara Universitesi Dis Hastanesi
   Bu isimlerin kisaltilmis, farkli yazilmis veya kismi halleri de dahildir (orn. "Medipol Mega", "Mega Hastanesi", "Medipol Universite Hastanesi").
   Genel olarak "Medipol" iceren tum hastane isimleri dahili sayilir.

   Yukaridaki listeye dahil OLMAYAN tum hastaneler harici (external) hastanelerdir.
   Bunlari [hospital_e1], [hospital_e2], ... [hospital_en] olarak tokenla.
## 4) Accesssion numaralari.
    Raporlara ozel, genellikle benzersiz olan bu numaralari [accession_1], [accession_2], ... [accession_n] olarak tokenla.

## 5) KORUNACAK BILGILER (DEGISTIRME):
   - Yas ve cinsiyet
   - Tum tibbi icerik (tani, tedavi, bulgular, olcumler, ilaclar, prosedurler)
   - Sehir / ilce / mahalle / semt isimleri
   - Cihaz ve marka isimleri
   - Protokol bilgileri
   - Ilac isimleri ve dozlar
   Bu korunacak bilgileri degistirmene gerek yok, oldugu gibi metinde birakabilirsin.

## 5) CIKTI FORMAT GEREKSINIMLERI:
   Ciktin TAM OLARAK iki bolumden olusmalidir ve bu iki bolum "|||MAPPING|||" ayraciyla ayrilmalidir.

   BOLUM 1: Anonimlestirilmis metin (yukaridaki kurallara gore tokenlenmis hali)

   AYRAC: |||MAPPING|||

   BOLUM 2: Kullanilan tum tokenlarin orijinal degerlerini gosteren bir JSON nesnesi.
   JSON formati:
   {
     "[token]": "orijinal deger",
     ...
   }

   - Girdi metnine hicbir ek kelime, aciklama veya yorum EKLEME.
   - Paragraf yapisi, bosluklar ve diger tum icerik oldugu gibi korunmalidir.
   - Metni sadece BIR KERE gonder, tekrarlama.

# ORNEKLER:

Girdi:
"Tetkik Dr. Ahmet Yilmaz tarafindan yapilmistir. Hasta Ayse Kara, 45 yasinda kadin. Tarih: 15.03.2024. Medipol Mega Universite Hastanesi."

Cikti:
Tetkik [radiologist_1] tarafindan yapilmistir. Hasta [patient_1], 45 yasinda kadin. Tarih: [date_1]. [hospital_1].
|||MAPPING|||
{"[radiologist_1]": "Dr. Ahmet Yilmaz", "[patient_1]": "Ayse Kara", "[date_1]": "15.03.2024", "[hospital_1]": "Medipol Mega Universite Hastanesi"}

---

Girdi:
"Prof. Dr. Mehmet Demir raporunu hazirlamistir. Hasta daha once 10.01.2023 tarihinde Sisli Etfal Hastanesi'nde gorulmus. Kontrol: 20.05.2024, Medipol Kosuyolu Hastanesi."

Cikti:
[radiologist_1] raporunu hazirlamistir. Hasta daha once [date_1] tarihinde [hospital_e1]'nde gorulmus. Kontrol: [date_2], [hospital_1].
|||MAPPING|||
{"[radiologist_1]": "Prof. Dr. Mehmet Demir", "[date_1]": "10.01.2023", "[hospital_e1]": "Sisli Etfal Hastanesi", "[date_2]": "20.05.2024", "[hospital_1]": "Medipol Kosuyolu Hastanesi"}

---

Girdi:
"SONUC: Sol minimal hidroureteronefroz, sol alt ureter tasi.                                                                                             Uzm. Dr. M. Kemal Ozkan                                                                                           Radyolog"

Cikti:
SONUC: Sol minimal hidroureteronefroz, sol alt ureter tasi.                                                                                             [radiologist_1]
|||MAPPING|||
{"[radiologist_1]": "Uzm. Dr. M. Kemal Ozkan, Radyolog"}
"""

print(f"[Rank {RANK}] Preparing prompts...")

reports_list = df["RaporText"].tolist()

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{report}"
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    for report in reports_list
]

# =======================================================================
# 4. Batched Generation with vLLM
# =======================================================================
MAX_NEW = 30000
CHUNK_SIZE = 5000

print(f"[Rank {RANK}] Running anonymization ({len(df)} reports)...")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW
)

# =======================================================================
# 4b. Parse outputs
# =======================================================================
MAPPING_DELIMITER = "|||MAPPING|||"


def extract_anonymized_and_mapping(text):
    text = text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    prefixes_to_remove = [
        "Iste anonimlestirilmis metin:",
        "Anonimlestirilmis metin:",
        "Cikti:",
        "Iste cikti:",
        "Anonimlestirilmis rapor:",
        "Sonuc:",
        "Here is the anonymized text:",
        "Anonymized text:",
        "Output:",
    ]
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    anonymized_text = text
    mapping_dict = {}
    mapping_json = "{}"

    if MAPPING_DELIMITER in text:
        parts = text.split(MAPPING_DELIMITER, 1)
        anonymized_text = parts[0].strip()
        raw_mapping = parts[1].strip()

        try:
            json_start = raw_mapping.find("{")
            json_end = raw_mapping.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = raw_mapping[json_start:json_end]
                mapping_dict = json.loads(json_str)
                mapping_json = json.dumps(mapping_dict, ensure_ascii=False)
        except json.JSONDecodeError:
            mapping_json = raw_mapping
            mapping_dict = {}

    return anonymized_text, mapping_dict, mapping_json


# =======================================================================
# 5. Process in chunks with checkpointing
# =======================================================================
save_cols = ["AccessionNo", "UID", "Batch", "KabulTarihi", "TetkikAdi",
             "RaporText", "Anonymized_Rapor", "Token_Mapping"]
save_cols = [c for c in save_cols if c in df.columns or c in ("Anonymized_Rapor", "Token_Mapping")]

total_processed = 0

for chunk_start in range(0, len(df), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
    chunk_df = df.iloc[chunk_start:chunk_end].copy()
    chunk_prompts = prompts[chunk_start:chunk_end]

    print(f"[Rank {RANK}] Processing chunk {chunk_start}-{chunk_end} / {len(df)}...")
    outputs = llm.generate(chunk_prompts, sampling_params)

    all_anonymized = []
    all_mappings_json = []
    mapping_rows = []

    for i, output in enumerate(outputs):
        anon_text, m_dict, m_json = extract_anonymized_and_mapping(output.outputs[0].text)
        all_anonymized.append(anon_text)
        all_mappings_json.append(m_json)

        accession = chunk_df.iloc[i]["AccessionNo"]
        if isinstance(m_dict, dict) and m_dict:
            for token, original in m_dict.items():
                mapping_rows.append({
                    "AccessionNo": accession,
                    "token": token,
                    "original_value": original
                })
        else:
            mapping_rows.append({
                "AccessionNo": accession,
                "token": "NO_MAPPING_EXTRACTED",
                "original_value": ""
            })

    chunk_df["Anonymized_Rapor"] = all_anonymized
    chunk_df["Token_Mapping"] = all_mappings_json

    result_cols = [c for c in save_cols if c in chunk_df.columns]
    result_chunk = chunk_df[result_cols]

    write_header = not os.path.exists(output_file)
    result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

    mapping_chunk_df = pd.DataFrame(mapping_rows)
    write_header_map = not os.path.exists(mapping_file)
    mapping_chunk_df.to_csv(mapping_file, mode="a", index=False, header=write_header_map, encoding="utf-8-sig")

    total_processed += len(chunk_df)
    print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

print(f"[Rank {RANK}] Done. Processed {total_processed} reports -> {output_file}")
