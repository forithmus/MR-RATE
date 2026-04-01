# Radiology Report Generation Pipeline

End-to-end pipeline for processing Turkish brain MRI radiology reports: anonymization, Turkish-to-English translation, quality control, and structured section extraction.

## Dataset

- **Source**: ~98,200 Turkish brain MRI radiology reports (2014-2020)
- **Batches**: batch00-batch27 across three collection periods
- **Model**: Qwen3.5-35B-A3B-FP8 via vLLM for all LLM steps

## Pipeline Overview

Each step follows an iterative **run → QC → retry → re-QC** loop until quality thresholds are met.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW TURKISH REPORTS                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  01_ANONYMIZATION                                               │
│  ├─ anonymize_reports_parallel.py                               │
│  │   Replace patient names, dates, hospitals, radiologist       │
│  │   names with tokens: [patient_1], [date_1], etc.             │
│  └─ QC: validate_anonymization_parallel.py (utils/)             │
│      Verify no PHI leakage, token consistency                   │
│      ↻ Re-anonymize failures                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  02_TRANSLATION                                                 │
│  ├─ translate_reports_parallel.py                               │
│  │   Turkish → English translation preserving medical           │
│  │   terminology, anonymization tokens, and report structure    │
│  └─ QC: → step 03                                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  03_TRANSLATION_QC                                              │
│  ├─ quality_check_parallel.py                                   │
│  │   LLM-based QC: checks translation completeness,             │
│  │   medical accuracy, token preservation                       │
│  ├─ detect_turkish_parallel.py                                  │
│  │   Detect remaining Turkish text in translations              │
│  ├─ retranslate_parallel.py                                     │
│  │   Re-translate QC failures with adjusted parameters          │
│  └─ ↻ Repeat QC → retranslate until <1% failure rate            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  04_STRUCTURING                                                 │
│  ├─ structure_reports_parallel.py  (with thinking)              │
│  │   Extract 4 sections from each report:                       │
│  │   clinical_information, technique, findings, impression      │
│  │                                                              │
│  │   Formatting rules:                                          │
│  │   - Findings: paragraph sentences (no bullets)               │
│  │   - Impression: em-dash (—) bullet points                    │
│  │   - Multi-section: subsection titles (Cranial:, Cervical:)   │
│  │                                                              │
│  ├─ structure_nothink_parallel.py  (no-think fallback)          │
│  │   For reports where thinking mode causes token exhaustion    │
│  │   Uses enable_thinking=False in chat template                │
│  └─ ↻ Run → collect failures → retry with no-think → merge      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  05_STRUCTURE_QC                                                │
│  ├─ qc_llm_verify.py  (with thinking)                           │
│  │   LLM compares structured output vs raw report               │
│  │   Checks: missing content, hallucinations, wrong section     │
│  │                                                              │
│  ├─ qc_llm_verify_nothink.py  (no-think fallback)               │
│  │   For QC reports where thinking mode fails to parse          │
│  └─ ↻ QC → fix failures → re-QC until pass rate >98%            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STRUCTURED ENGLISH REPORTS                  │
│  Per-batch CSVs: batch{NN}_reports.csv                          │
│  Columns: Studiy_UID, report, clinical_information, technique,  │
│           findings, impression                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  06_PATHOLOGY_CLASSIFICATION                                    │
│  ├─ classify_pathologies_parallel.py                            │
│  │   3-step LLM classification per report:                      │
│  │     Step 1: CoT reasoning with exact quotes                  │
│  │     Step 2: JSON extraction with cross-validation            │
│  │     Step 3: Verification of PRESENT labels                   │
│  │   37 SNOMED CT-grounded pathologies, 0/1 binary labels       │
│  │                                                              │
│  ├─ merge_labels.py                                             │
│  │   Merge per-rank JSONs into single CSV                       │
│  │                                                              │
│  └─ data/                                                       │
│      ├─ pathologies.json          (pathology definitions)        │
│      └─ pathologies_snomed_map.json (SNOMED CT mappings)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PATHOLOGY LABELS CSV                           │
│  labels.csv                                                     │
│  Columns: study_uid, Cerebral infarction, Cerebral hemorrhage,  │
│           ... (37 pathology columns, values 0 or 1)             │
└─────────────────────────────────────────────────────────────────┘
```

## Step Details

### 01_anonymization

**Script**: `anonymize_reports_parallel.py`

Replaces protected health information with deterministic tokens:

| Entity | Token Format |
|--------|-------------|
| Patient names | `[patient_1]`, `[patient_2]` |
| Dates | `[date_1]`, `[date_2]` |
| Radiologist names | `[radiologist_1]` |
| Hospital names | `[hospital_1]` |
| Protocol numbers | `[protocol_1]` |
| Registration numbers | `[registration_1]` |
| Accession numbers | `[accession_1]` |

**QC**: `utils/validate_anonymization_parallel.py` — verifies no original PHI remains and tokens are consistent.

**Input**: Raw Turkish reports CSV
**Output**: Anonymized Turkish reports CSV with Token_Mapping column

### 02_translation

**Script**: `translate_reports_parallel.py`

- Model: Qwen3.5-35B-A3B-FP8 via vLLM
- Preserves anonymization tokens verbatim
- Preserves medical terminology (parasagittal, prepontine, cerebellopontine, etc.)
- Temperature: 0.1, max_tokens: 30000

**Input**: Anonymized Turkish reports CSV
**Output**: English translations CSV

### 03_translation_qc

Iterative QC loop with three tools:

1. **`quality_check_parallel.py`** — LLM compares Turkish source vs English translation for completeness, accuracy, and token preservation
2. **`detect_turkish_parallel.py`** — Rule-based detection of remaining Turkish words/phrases
3. **`retranslate_parallel.py`** — Re-translates reports that failed QC

**Loop**: Run QC → collect failures → retranslate → re-run QC → repeat until failure rate <1%

**Manual review**: Once the automated QC pass rate stabilizes (typically >98%), the remaining failures are reviewed manually. Many turn out to be **false positives** — the QC model flags issues that aren't actually errors (e.g., acceptable paraphrasing, intentional terminology choices). Genuine failures are fixed individually and re-verified.

### 04_structuring

Two-pass approach to handle model thinking limitations:

1. **`structure_reports_parallel.py`** (thinking enabled) — Primary pass. Most reports parse successfully with the model's chain-of-thought reasoning.
2. **`structure_nothink_parallel.py`** (thinking disabled) — Fallback for reports where the model exhausts tokens on thinking without producing output. Uses `enable_thinking=False` and `/no_think` prompt.

**Loop**: Run primary → collect parse failures → retry with no-think → merge results → repeat until 0 parse failures

**Manual review**: After automated retries reach 0 parse failures, rule-based QC checks for formatting issues (header leaks, bullet formatting, tabs, empty sections). Remaining issues are reviewed manually — many are false positives (e.g., "Technique:" appearing inside technique content is normal, not a header leak).

**Key formatting rules enforced**:
- Findings: flowing paragraph sentences, no bullet points
- Impression: em-dash (—) bullet points, one finding per bullet
- Multi-section reports (e.g., Brain + Cervical MRI): section titles (Cranial:, Cervical:, MR Angiography:)
- "Included in the examination" items: kept in both findings AND impression

### 05_structure_qc

LLM-based verification of structured output against raw reports:

1. **`qc_llm_verify.py`** (thinking enabled) — Checks for missing content, hallucinated content, and content in wrong section
2. **`qc_llm_verify_nothink.py`** (thinking disabled) — Fallback for QC parse failures

**Loop**: Run QC → fix failures → re-QC until pass rate >98%

**Manual review**: Once automated QC stabilizes, remaining failures are reviewed manually in batches. Common false positive patterns:
- "Comparison info in technique instead of separate section" — debatable placement, not an error
- "Incidental findings moved from impression to findings" — acceptable if content is preserved
- "Empty impression" — correct if the original report had no impression section (never synthesize)
- Laterality discrepancies between findings and impression — these exist in the original reports; findings are ground truth

Genuine failures (missing content, hallucinated content, wrong vascular territory) are fixed and re-verified until the manual review pass rate converges.

**What is NOT flagged** (acceptable):
- Formatting differences (bullets, whitespace)
- Section title additions (Cranial:, Cervical:)
- Comparison info placement (technique vs findings)

### 06_pathology_classification

**Script**: `classify_pathologies_parallel.py`

Multi-step LLM-based binary classification of structured reports against 37 brain/spine MRI pathologies, each grounded to SNOMED CT or RadLex.

**Three-step pipeline per report:**

1. **Step 1 — CoT reasoning**: The model reads the `findings` section and reasons through each of the 37 pathologies, quoting exact sentences from the report. Strict inference rules prevent cross-pathology contamination (e.g., "ischemic-gliotic lesions" → Gliosis=PRESENT but Cerebral infarction=ABSENT).

2. **Step 2 — JSON extraction**: The model converts its CoT analysis into a structured JSON with 0/1 values. Cross-validated against CoT labels — CoT is authoritative on disagreement. Failed JSON parses are retried up to 2 times, with CoT-only fallback.

3. **Step 3 — Verification**: For reports with any PRESENT labels, the model re-examines each one against explicit KEEP/ABSENT rules to catch false positives from invalid inference. Only 1→0 flips are allowed (removing false positives; no new PRESENT labels introduced).

**Pathologies**: 37 brain/spine MRI pathologies defined in `data/pathologies.json`. Each has positive/negative sentence pairs for contrastive pretraining. SNOMED CT/RadLex mappings in `data/pathologies_snomed_map.json`.

| # | Pathology | SNOMED CT ID | Prevalence |
|---|-----------|-------------|------------|
| 1 | Gliosis | — | 35.5% |
| 2 | Cerebral atrophy | 278849000 | 14.5% |
| 3 | Ventriculomegaly | 413808003 | 11.2% |
| 4 | Cerebral edema | 2032001 | 4.8% |
| 5 | Cerebral infarction | 432504007 | 3.8% |
| 6 | Empty sella syndrome | 237722004 | 3.8% |
| 7 | Arachnoid cyst | 33595009 | 3.6% |
| 8 | Metastatic malignant neoplasm to brain | 94225005 | 3.4% |
| 9 | Demyelinating disease of CNS | 6118003 | 3.0% |
| 10 | Mastoiditis | 52404001 | 2.9% |
| 11–37 | *(see `data/pathologies.json` for full list)* | | |

**Model**: Qwen3.5-35B-A3B via vLLM, temperature=0, seed=42

**Input**: Structured report CSVs from step 05 (`batch{NN}_reports.csv` with `study_uid` and `findings` columns)

**Output**: Per-rank `labels_rank_{RANK}.json` containing metadata, stats, and per-report results (labels + CoT reasoning)

**Merge**: `merge_labels.py` combines per-rank JSONs into a single `labels.csv` with `study_uid` + 37 pathology columns (0/1)

**Validation**: Manual review of 100 random samples showed 0% clear errors and ~4% borderline cases (findings-only evaluation).

**Run:**

```bash
# Classification (distributed across 8 nodes x 4 GPUs = 32 ranks)
srun python 06_pathology_classification/classify_pathologies_parallel.py \
    --reports_dir /path/to/reports \
    --pathologies_json 06_pathology_classification/data/pathologies.json \
    --output_dir /path/to/output \
    --model_name Qwen/Qwen3.5-35B-A3B \
    --batch_size 500 \
    --seed 42

# Merge per-rank results into single CSV
python 06_pathology_classification/merge_labels.py \
    --input_dir /path/to/output \
    --output /path/to/labels.csv
```

## QC

Every QC step follows the same iterative pattern:

```
Run LLM on full dataset
        │
        ▼
Run automated QC (LLM or rule-based)
        │
        ▼
Pass rate >98%? ──no──► Collect failures
        │                     │
       yes                    ▼
        │              Retry with adjusted params
        │              (no-think fallback, higher tokens)
        │                     │
        ▼                     └──► Re-run QC ──► loop
Manual review of remaining failures
        │
        ▼
Many are FALSE POSITIVES (QC model flagging
acceptable content as errors)
        │
        ▼
Fix genuine failures only
        │
        ▼
Re-verify fixes ──► more false positives filtered
        │
        ▼
Converged ✓
```

**Note**: The QC model is stricter than necessary. At each round, a significant portion of "failures" are false positives — the QC model flags debatable issues (comparison placement, incidental findings organization) as errors. Manual review filters these out, and only genuine content issues (missing findings, hallucinations, wrong laterality) are fixed.

**Typical convergence**: 85% pass after first run → 98%+ after retries → manual review of remaining 1-2% reveals ~50% false positives → fix genuine issues → re-verify → done.

## Parallel Execution

All scripts use SLURM-based data parallelism:

```python
RANK = int(os.environ.get("SLURM_PROCID", "0"))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))
df = df.iloc[RANK::WORLD_SIZE]  # Shard by rank
```

Each rank:
- Loads its own vLLM engine on one GPU
- Processes its shard independently
- Writes to `output_dir/{prefix}_rank_{RANK}.csv`
- Supports resume via AccessionNo deduplication

Merge with: `utils/merge_shards.py --shard_dir <dir> --output <file>`

## Output Format

Final structured reports per batch:

| Column | Description |
|--------|-------------|
| `UID` | Unique study identifier |
| `report` | Full English translated report |
| `clinical_information` | Clinical indication/history (empty if not present) |
| `technique` | Imaging technique and sequences |
| `findings` | All observations as paragraph sentences |
| `impression` | Conclusions as em-dash bullet points |

