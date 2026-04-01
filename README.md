# **MR-RATE: A Vision-Language Foundation Model and Dataset for Magnetic Resonance Imaging**

Welcome to the official repository for MR-RATE, a pioneering vision-language model and 3D medical imaging dataset for Magnetic Resonance Imaging.

**🔗 Resources:**

- Paper: **coming soon**
- Model Weights: **coming soon**
- Dataset: **[MR-RATE on Hugging Face](https://huggingface.co/datasets/Forithmus/MR-RATE)**

---

## 🧠 Overview

**MR-RATE** is a unified framework for **vision-language modeling in brain and spine MRI**, comprising a large-scale 3D medical imaging dataset that uniquely pairs textual data with brain and spine MRI volumes and a contrastive pretraining pipeline that aligns multi-sequence MRI volumes with radiology reports using VL-CABS loss.

---

## 📁 Repository Structure

```
MR-RATE/
│
├── data-preprocessing/       # Data preprocessing pipeline and dataset download scripts
│
├── contrastive-pretraining/  # Contrastive pretraining code for vision-language modeling
│
└── README.md
```

Each folder includes its own `README.md` detailing configuration, dependencies, and usage.

---

## 🛠️ Components


| Component                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Preprocessing**      | End-to-end pipeline converting raw DICOM exports into clean, anonymized, and spatially standardized NIfTI volumes. Covers DICOM-to-NIfTI conversion, PACS metadata filtering, modality classification, brain segmentation & defacing, co-registration to a shared T1w reference, and atlas registration to MNI152 space. Also includes the radiology report preprocessing pipeline (anonymization, translation, structuring, QC) and LLM-based pathology classification producing binary labels for 37 SNOMED CT-grounded pathologies. Includes standalone scripts for downloading and merging all MR-RATE Hugging Face repositories. |
| **Contrastive Pretraining** | Contrastive vision-language model that aligns multi-sequence MRI volumes and radiology reports using VL-CABS loss. Uses a VJEPA2 (ViT-G) image encoder with LoRA fine-tuning and a BiomedVLP-CXR-BERT text encoder. Supports four multi-volume fusion modes (`early`, `mid_cnn`, `late`, `late_attn`) and three image spaces (`native_space`, `coreg_space`, `atlas_space`). Enables zero-shot brain MRI pathology classification at inference time. |


---

## 🧩 Workflow Summary

1. **Download preprocessed data** directly from [MR-RATE on Hugging Face](https://huggingface.co/datasets/Forithmus/MR-RATE) (see `data-preprocessing/` for preprocessing details)
2. **Train the contrastive model** on (multi-sequence MRI, radiology report) pairs using `contrastive-pretraining/`
4. **Run zero-shot inference** for brain MRI pathology classification using trained model weights and `contrastive-pretraining/`

---

## Citation

If you use this repository, the dataset, or any of its components, please cite:

```
Coming soon
```

---

## License

We are committed to fostering innovation and collaboration in the research community. All elements of the MR-RATE repository are released under the **[Creative Commons Attribution–NonCommercial–ShareAlike (CC BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license.

This allows all elements to be freely used, modified, and shared for **non-commercial research purposes**, provided that the original work is properly cited and any derivative works are distributed under the same license.

For commercial inquiries related to MR-RATE, please contact: contact@forithmus.com