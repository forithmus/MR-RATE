# =======================================================================
# Parallel Retranslation of non-English reports
# Takes turkish_anonymized_report and translates to English
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
    parser.add_argument("--reports_file", type=str, required=True,
                        help="Final anonymized reports CSV")
    parser.add_argument("--detection_file", type=str, default=None,
                        help="Language detection CSV (if None, translate ALL reports)")
    parser.add_argument("--output_dir", type=str, default="retranslate_shards")
    args = parser.parse_args()

    RANK = int(os.environ.get("SLURM_PROCID", "0"))
    WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))

    print(f"[Rank {RANK}/{WORLD_SIZE}] Starting on node {os.environ.get('SLURMD_NODENAME', 'local')}")

    # =======================================================================
    # 1. Load reports
    # =======================================================================
    print(f"[Rank {RANK}] Loading reports...")

    reports = pd.read_csv(args.reports_file, encoding='utf-8-sig')
    reports["AccessionNo"] = reports["AccessionNo"].astype(str)

    # Normalize column names for 2017-2019 dataset compatibility
    if "Anonymized_Rapor" in reports.columns and "turkish_anonymized_report" not in reports.columns:
        reports.rename(columns={"Anonymized_Rapor": "turkish_anonymized_report"}, inplace=True)
    if "Batch" in reports.columns and "batch_number" not in reports.columns:
        reports.rename(columns={"Batch": "batch_number"}, inplace=True)

    if args.detection_file:
        detection = pd.read_csv(args.detection_file)
        non_english = detection[detection["detected_language"] != "english"]["AccessionNo"].astype(str)
        df = reports[reports["AccessionNo"].isin(non_english)].reset_index(drop=True)
        print(f"[Rank {RANK}] Filtered to {len(df)} non-English reports")
    else:
        df = reports.copy()
        print(f"[Rank {RANK}] Translating ALL {len(df)} reports")

    # Shard
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"retranslate_rank_{RANK}.csv")

    df = df.iloc[RANK::WORLD_SIZE].reset_index(drop=True)
    print(f"[Rank {RANK}] This shard: {len(df)}")

    # Resume support
    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file, usecols=["AccessionNo"])
        done_accessions = set(done_df["AccessionNo"].astype(str))
        mask = ~df["AccessionNo"].astype(str).isin(done_accessions)
        df = df[mask].reset_index(drop=True)
        print(f"[Rank {RANK}] Resuming: {len(done_accessions)} already done, {len(df)} remaining.")

    if len(df) == 0:
        print(f"[Rank {RANK}] All reports already retranslated.")
        exit(0)

    # =======================================================================
    # 2. Load Model with vLLM
    # =======================================================================
    model_dir = "Qwen/Qwen3.5-35B-A3B-FP8"

    print(f"[Rank {RANK}] Loading model with vLLM engine...")
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print(f"[Rank {RANK}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    print(f"[Rank {RANK}] vLLM engine and tokenizer loaded.")

    # =======================================================================
    # 3. System Prompt
    # =======================================================================
    system_prompt = """You are a Turkish-to-English medical translator specializing in brain MRI radiology reports. Your ONLY job is to translate Turkish text into US medical English. Every single Turkish word MUST be converted to standard US medical English as used in American radiology practice.

    CRITICAL RULES:
    - The input is ALWAYS in Turkish. You MUST translate it to English. Do NOT copy the Turkish text.
    - Translate ALL Turkish words including headers, section names, and descriptions.
    - Keep placeholder tokens like [patient_1], [date_1], [doctor_1], [hospital_1], [radiologist_1] exactly as they are.
    - Preserve the structure and formatting (sections, line breaks, bullet points).
    - Output ONLY the English translation, nothing else.

    SECTION HEADERS - translate these exactly:
    - "Bulgular" / "BULGULAR" -> "Findings"
    - "Sonuç" / "SONUÇ" / "Yorum" / "Değerlendirme" -> "Impression"
    - "Teknik" / "İnceleme tekniği" -> "Technique"
    - "Klinik bilgi" / "Klinik" / "Ön tanı" / "Endikasyon" -> "Clinical information"

    REPORT TITLES - translate these exactly:
    - "BEYİN MRG" / "KRANİAL MRG" -> "BRAIN MRI" / "CRANIAL MRI"
    - "MRG" -> "MRI", "BT" -> "CT"
    - "KONTRASTSIZ" / "KONTRASTLI" -> "NON-CONTRAST" / "CONTRAST-ENHANCED"
    - "SERVİKAL VERTEBRA" -> "CERVICAL VERTEBRA"

    COMMON ANATOMICAL TERMS:
    - "beyaz cevher" -> "white matter", "gri cevher" -> "gray matter"
    - "bazal ganglionlar" -> "basal ganglia", "lateral ventriküller" -> "lateral ventricles"
    - "serebellar hemisfer" -> "cerebellar hemisphere", "serebellar folia" -> "cerebellar folia"
    - "beyin sapı" -> "brainstem", "korpus kallozum" -> "corpus callosum"
    - "sentrum semiovale" -> "centrum semiovale", "korona radiyata" -> "corona radiata"
    - "internal kapsül" -> "internal capsule", "eksternal kapsül" -> "external capsule"
    - "optik kiazma" -> "optic chiasm", "hipofiz bezi" -> "pituitary gland"
    - "koroid pleksus" -> "choroid plexus", "pineal bez" -> "pineal gland"
    - "kortikal sulkus" -> "cortical sulcus", "serebral parankım" -> "cerebral parenchyma"
    - "kafa çiftleri" / "sinir çiftleri" -> "cranial nerves"
    - "sisterna magna" -> "cisterna magna", "sella tursika" -> "sella turcica"
    - "yer kaplayıcı lezyon" -> "space-occupying lesion"
    - "periventriküler" -> "periventricular" (NEVER translate as "subcortical")
    - "supratentoryal" -> "supratentorial", "infratentoryal" -> "infratentorial"
    - "intrakraniyel" -> "intracranial"
    - "unkus" -> "uncus" (NOT "temporal lobe")
    - "falks serebri" -> "falx cerebri", "falks serebelli" -> "falx cerebelli" (NOT "falcine")
    - "tetraventriküler" -> "tetraventricular" / "quadriventricular" (refers to ALL FOUR VENTRICLES, NOT quadrigeminal cistern)
    - "kranioservikal" -> "craniocervical" (NOT "craniospinal")

    CRITICAL ANATOMICAL DISTINCTIONS - DO NOT CONFUSE:
    - "bulbus" in brainstem structure lists (e.g., "bulbus, pons, mezensefalon") -> "medulla oblongata" (NOT "midbrain", "olfactory bulb", "putamen", "pons", or "bulb"). "bulbus" in brainstem context always means "medulla oblongata".
    - "bulbus oculi" (eyeball context) -> "globe" or "globus oculi"
    - "PET-BT" -> "PET-CT" (BT is the Turkish abbreviation for CT, always translate to CT)
    - "aynı" -> "same" (NOT "similar" — "aynı" means identical, not approximate)
    - "orbitalar" -> "orbits" (NOT "orbitals")
    - "spondilotik" -> "spondylotic" (NOT "spondyloitic")
    - "ekstraaksiyel" -> "extra-axial" (NOT "extraxially" or "extraaxial")
    - "hemosiderin" -> "hemosiderin" (keep as is — do NOT translate as "hemorrhagic". Hemosiderin = chronic iron deposit, hemorrhagic = active bleeding)
    - "çıkartılmış" -> "removed" / "extracted" (NOT "placed" or "previously placed" — opposite meaning)
    - "kapatmış" / "kapatmak" -> "covered" / "occluded" (NOT "compressed" / "compressing")
    - "birer adet" -> "one each" (distributive — "birer adet" with two locations = TWO separate foci, one in each)
    - "sekel" -> "sequela" (plural: "sequelae") — chronic residual finding from a prior event
    - "orbitalar" -> "orbits" (eye sockets, NOT "orbitals"), "bulbus okuli" -> "globes" (eyeballs — different from orbits)
    - "supraserebellar" -> "supracerebellar" (above the cerebellum) — NEVER translate as "suprasellar" (above the sella). These are DIFFERENT anatomical locations!
    - "suprasellar" -> "suprasellar" (above the sella turcica) — keep as is
    - "Willis poligonu" -> "circle of Willis" (NOT "Willis polygon" — the standard English medical term is "circle of Willis")
    - "milimetrik" -> "millimetric" / "millimeter-sized" (NOT "mild" — milimetrik refers to SIZE, not severity)
    - "sinyalsiz" -> "flow void" (in vascular context, e.g., arteries and venous sinuses) or "signal void" (in non-vascular context) — NOT "non-enhancing". "sinyalsiz" means absence of signal, not absence of enhancement
    - "serebellopontin köşe" -> "cerebellopontine angle" (NOT "sylvian" — completely different anatomical location)
    - "serebellopontin köşe sisternleri" -> "cerebellopontine angle cisterns" (NOT "sylvian cisterns")
    - "perimezensefalik" -> "perimesencephalic" (NOT "permesencephalic" — correct spelling)
    - "vermis" / "vermiş" -> "vermis" (cerebellar vermis — NOT "brainstem". "vermiş" is sometimes a Turkish typo for "vermis")
    - "kanaliküler" -> "canalicular" (relating to the canal/petrous bone — NOT "intracranial")
    - "retroserebellar" -> "retrocerebellar" (one word, NOT "retros cerebellar" or "retro cerebellar")
    - "mastikatuar alan" -> "masticatory space" (NOT "masseter" — masticatory space is the anatomical compartment, masseter is just one muscle within it)
    - "aerasyon" -> "aeration" (air content — NOT "opacification", which means the OPPOSITE)
    - "klinoid" -> "clinoid" (NOT "clival" — clinoid process vs clivus are different structures)
    - "supraklinoid" -> "supraclinoid" (NOT "supraclival")
    - "kraniofasyal" -> "craniofacial" (NOT "craniomaxillary")
    - "vazojenik ödem" -> "vasogenic edema" (NOT "vascular edema" — vasogenic is the correct medical term)
    - "forniksler" -> "fornices" (NOT "fimbriae" — fornix/fornices are brain structures, fimbriae are different)
    - "3. ventrikül" / "üçüncü ventrikül" -> "third ventricle" (when "3. ve lateral ventriküller" appears, translate as "third and lateral ventricles" — do NOT drop the "third")
    - "Akc" -> "lung" (abbreviation for "akciğer")
    - "opere" -> "operated" / "post-surgical"

    COMMON IMAGING TERMS:
    - "aksiyel" -> "axial", "koronal" -> "coronal", "sagittal" -> "sagittal"
    - "çok düzlemli" -> "multiplanar"
    - "çok sekanslı" -> "multi-sequence" (NOT "multislice")
    - "DAG" -> "DWI" (diffusion-weighted imaging)
    - "IVKM" -> "IV contrast agent"
    - "kontrast madde" -> "contrast agent", "kontrast tutulumu" -> "contrast enhancement"
    - "kontrastlanma" -> "enhancement", "difüzyon kısıtlaması" -> "diffusion restriction"
    - "T1A" -> "T1-weighted" or keep as "T1A", "T2A" -> "T2-weighted" or keep as "T2A" (do NOT drop the "A")
    - "Hemo" / "HEMO" -> "SWI" or "hemorrhage-sensitive sequence" (NOT "Hemos")
    - "TSE" -> "TSE" (turbo spin echo), "FFE" -> "FFE" (fast field echo)

    COMMON FINDING TERMS:
    - "hiperintens" -> "hyperintense", "hipointens" -> "hypointense", "izointens" -> "isointense"
    - "gliotik" -> "gliotic", "iskemik" -> "ischemic", "hemorajik" -> "hemorrhagic"
    - "sinyal değişikliği" -> "signal change", "sinyal artışı" -> "signal increase"
    - "patolojik" -> "pathological", "nonspesifik" -> "nonspecific"
    - "hafif" -> "mild", "belirgin" -> "prominent" / "marked"
    - "genişleme" -> "enlargement" / "widening" (translate literally, do NOT interpret as "hydrocephalus" unless the Turkish text says "hidrosefali")

    COMMON VERB ENDINGS:
    - "doğaldır" / "normaldir" -> "is normal"
    - "izlenmiştir" / "izlenmektedir" -> "is observed"
    - "izlenmemiştir" -> "is not observed"
    - "saptanmamıştır" -> "was not detected"
    - "saptanmıştır" -> "was detected"
    - "mevcuttur" -> "is present"
    - "gözlenmemiştir" -> "was not observed"
    - "değerlendirilmiştir" -> "was evaluated"

    CRITICAL TRANSLATION RULES:
    - Do NOT add words that are not in the original Turkish text (no hallucination)
    - Do NOT add severity modifiers (mild, severe) unless explicitly in the Turkish
    - Do NOT add anatomical side (right/left) unless explicitly stated in Turkish
    - Do NOT interpret or diagnose — translate literally and faithfully
    - Preserve ALL measurements exactly as written (mm, cm, numbers)

    CLOSING PHRASES - do NOT include these in the translation:
    - "Saygılarımla" / "Saygılarımızla" (Sincerely)
    - "Sayın Meslektaşım" (Dear Colleague)"""

    # =======================================================================
    # 4. Batched Processing
    # =======================================================================
    CHUNK_SIZE = 2000

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=49152,
    )

    total_processed = 0

    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        turkish_reports = chunk_df["turkish_anonymized_report"].tolist()

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Translate this Turkish radiology report to English:\n\n{report if isinstance(report, str) else '[EMPTY]'}",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for report in turkish_reports
        ]

        print(f"[Rank {RANK}] Translating chunk {chunk_start}-{chunk_end} / {len(df)}...")
        outputs = llm.generate(prompts, sampling_params)

        rows = []
        retry_indices = []
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()

            # Strip thinking tags if present
            translation = raw
            if "</think>" in translation:
                translation = translation[translation.rfind("</think>") + len("</think>"):].strip()

            # Check if translation is still Turkish (failed)
            still_turkish = any(w in translation for w in [
                "izlenmiştir", "izlenmemiştir", "saptanmamıştır", "doğaldır",
                "normaldir", "mevcuttur", "mektedir", "maktadır", "mıştır",
                "Bulgular:", "Klinik bilgi:", "BULGULAR", "SONUÇ",
            ])

            if still_turkish:
                retry_indices.append(i)
            else:
                rows.append({
                    "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
                    "UID": chunk_df.iloc[i]["UID"],
                    "batch_number": chunk_df.iloc[i]["batch_number"],
                    "english_translation": translation,
                })

        # Retry failed translations with a more forceful prompt
        if retry_indices:
            print(f"[Rank {RANK}] Retrying {len(retry_indices)} failed translations...")
            retry_prompts = []
            for i in retry_indices:
                report = turkish_reports[i] if isinstance(turkish_reports[i], str) else "[EMPTY]"
                retry_prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Translate this Turkish radiology report to English. Do NOT output Turkish. Every Turkish word must become English:\n\n{report}",
                            },
                            {
                                "role": "assistant",
                                "content": "<think>\nI need to translate this Turkish medical report into English. Let me translate every sentence.\n</think>\n\n",
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )

            retry_params = SamplingParams(temperature=0.6, max_tokens=24576)
            retry_outputs = llm.generate(retry_prompts, retry_params)

            for j, output in enumerate(retry_outputs):
                i = retry_indices[j]
                raw = output.outputs[0].text.strip()
                translation = raw
                if "</think>" in translation:
                    translation = translation[translation.rfind("</think>") + len("</think>"):].strip()

                rows.append({
                    "AccessionNo": chunk_df.iloc[i]["AccessionNo"],
                    "UID": chunk_df.iloc[i]["UID"],
                    "batch_number": chunk_df.iloc[i]["batch_number"],
                    "english_translation": translation,
                })

            print(f"[Rank {RANK}] Retry complete.")

        result_chunk = pd.DataFrame(rows)

        write_header = not os.path.exists(output_file)
        result_chunk.to_csv(output_file, mode="a", index=False, header=write_header, encoding="utf-8-sig")

        total_processed += len(chunk_df)
        print(f"[Rank {RANK}] Saved chunk ({total_processed}/{len(df)} total).")

    print(f"[Rank {RANK}] Done. Retranslated {total_processed} reports -> {output_file}")
