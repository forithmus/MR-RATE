# MR-RATE report writer (FORA recipe)

Brain MRI report generation on top of the 74-diagnosis MR-RATE MIL (`mrdino3d_hplus_atlas_mil_epoch004`, ViT-H+ DINO,
validation macro AUROC 0.886), with the writer of FORA
([github.com/forithmus/FORA](https://github.com/forithmus/FORA), `report-generation/`). The frozen MIL reads every
sequence of a study. Its calibrated predictions become a diagnostic plan, and a Gemma-4-31B-it writer with LoRA reads
the MIL tokens of the whole study together with the plan and writes the Findings section. Training is SFT followed by
offline DPO.

## Writer input

| part | content |
|---|---|
| visual prefix | every MIL token of every sequence (the 1304-d classifier input: normalised last-block feature + position features), with a sequence-kind (T1, T2, FLAIR, SWI, ...) and sequence-ordinal embedding, resampled to 512 latents (depth 2, 8 heads, width 1024), plus a study-type token |
| label tokens | one token per diagnosis from its name and frozen Platt-calibrated probability |
| plan | sequence list, then confident and possible findings (FORA tiers, train-only calibration, at most 20 possible findings) |
| per planned finding | calibrated probability and base rate, an evidence description (distribution, main and further sites as brain regions with side, sequences holding the evidence), the attention-pooled feature and up to 16 focus tokens |

The evidence of a finding is the positive part of attention x token logit x class scale over all tokens of the study.
It is summed over sequences on the shared atlas grid, smoothed (4 mm), cut to the region holding 50 % of the mass and
split into foci. Region names come from the NV-Segment-CTMR brain parcellation in MR-RATE atlas space (`assets/`,
`regions.py`). Reference labels and report text never enter the writer input.

## Pipeline

| stage | module | what it does |
|---|---|---|
| 1 | `mrrate_report.predict` | frozen MIL logits of every study (train, val, test), resumable shards; validation studies are checked against the MIL run's own predictions |
| 2 | `mrrate_report.calibrate` | train-only Platt and Youden tiers (FORA rules), calibration JSON and dense logits |
| 3 | `mrrate_report.train_sft` | SFT, Gemma-4-31B-it + LoRA r16/a32 under FSDP2, one study per GPU per step, MIL tokens computed online |
| 4 | `mrrate_report.tag_sentences` | Gemma (vLLM) tags every sentence of every train report with the diagnoses it states as present, possible or absent |
| 5 | `mrrate_report.dpo_pairs` | rejected reports by rule: omission, multi-omission, insertion of a planned or random diagnosis by a donor sentence, normalisation, repetition |
| 6 | `mrrate_report.train_dpo` | offline DPO from the SFT checkpoint (beta 0.1, two pairs per visit, chosen CE) |
| 7 | `mrrate_report.generate` | greedy Findings for a split, CSV `study_uid, findings_gt, findings_pred, ...` |
| 8 | `report-generation-training` evaluation | the repository's 74-label extractor and clinical / NLG metrics on the generated CSV |

`scripts/dfw/` holds the launchers (`run_stage.sbatch`, one stage per job, resumable). See the commands there.

## Requirements

The MIL bundle environment (PyTorch, nibabel, the pinned DINOv3 checkout), `transformers` with Gemma-4, `peft`,
`scipy`, a FORA checkout with `fora` installed and `report-generation/` on `PYTHONPATH` (FSDP and DPO utilities), and
vLLM for stage 4.

## Tests

```bash
python -m pytest tests        # CPU: evidence foci and tokens, calibration, plan text, sentence tags, DPO pairs
```
