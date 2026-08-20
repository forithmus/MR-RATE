# NeuroVFM MRI diagnosis extraction

This is the report-to-label pipeline used for the current MR-RATE clinical
evaluation target. It classifies each report against the 74 expert-defined
NeuroVFM MRI diagnoses and returns a binary label plus a short rationale for
every diagnosis.

The checked-in prompt, diagnosis order, guidance, JSON parser, and inference
defaults match the pipeline used for the completed MR-RATE challenge
re-extraction. The older `06_pathology_classification` directory remains in
the repository only to reproduce the historical 37-pathology labels.

Run one or more Slurm ranks with:

```bash
srun --gpu-bind=none python extract_neurovfm_dx_gemma.py \
  --reports_dir /path/to/reports \
  --diagnoses_json data/neurovfm_mri_diagnoses.json \
  --output_dir /path/to/shards

python merge_labels.py \
  --input_dir /path/to/shards \
  --output /path/to/neurovfm74_labels.csv \
  --rationales /path/to/neurovfm74_rationales.json
```

Input files are named `batchNN_reports.csv` and require `study_uid`. The
extractor uses `findings`, `impression`, and `clinical_information`, falling
back to `report` when the structured result sections are empty. Inference is
deterministic (`temperature=0`, seed 42), checkpoints after each batch, and
resumes globally by `study_uid` across rank files.
