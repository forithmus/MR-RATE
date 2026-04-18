"""
Inference dataset for MR-RATE.

Mirrors the training preprocessing from data.py:
  - RAS reorientation (nib.as_closest_canonical)
  - Target spacing (1.0, 0.5, 0.5) and target shape (256, 384, 384)
  - Posterior shift (15 mm) on W axis to compensate for defacing
  - Same layout auto-detection (<space>/img vs batchXX/<uid>/img)

Differs from training only in inference semantics:
  - Deterministic (no random sentence sampling, no truncation)
  - Returns subject_id for result tracking
  - Optional per-subject labels loaded from CSV

Returns: (images, sentences, subject_id, real_volume_mask, labels)
  - images:            [N, 1, D, H, W]  variable N volumes
  - sentences:         list of report sentences
  - subject_id:        str
  - real_volume_mask:  [N] boolean, all True (no padding at batch_size=1)
  - labels:            np.ndarray (empty if labels_file not provided)
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset

from data import NORMALIZERS, resize_array


class MRReportDatasetInfer(Dataset):
    """
    Inference dataset for brain MRI with variable volumes per subject.

    Preprocessing is kept identical to MRReportDataset (data.py) so the
    model sees distributionally matched inputs at train and inference time.
    """

    def __init__(
        self,
        data_folder,
        jsonl_file,
        target_spacing=(1.0, 0.5, 0.5),
        target_shape=(256, 384, 384),
        posterior_shift_mm=15.0,
        space="native_space",
        normalizer="zscore",
        normalizer_kwargs=None,
        labels_file=None,
        splits_csv=None,
        split="test",
    ):
        self.data_folder = data_folder
        self.space = space
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.posterior_shift_voxels = int(round(posterior_shift_mm / target_spacing[2]))

        if normalizer not in NORMALIZERS:
            raise ValueError(
                f"Unknown normalizer '{normalizer}'. "
                f"Choose from: {list(NORMALIZERS.keys())}"
            )
        normalizer_kwargs = normalizer_kwargs or {}
        self.normalizer_obj = NORMALIZERS[normalizer](**normalizer_kwargs)

        self.split_uids = self._load_splits(splits_csv, split) if splits_csv else None

        self.subject_to_sentences = self._load_jsonl(jsonl_file)

        self.subject_to_labels = {}
        self.label_columns = []
        if labels_file is not None:
            self._load_labels(labels_file)

        self.samples = self._prepare_samples(data_folder)

        print(f"[MRReportDatasetInfer] Found {len(self.samples)} subjects "
              f"(space={space}, normalizer={normalizer})")
        if self.label_columns:
            print(f"[MRReportDatasetInfer] Labels loaded: {len(self.label_columns)} classes")

    @staticmethod
    def _load_splits(splits_csv, split):
        """Load study UIDs belonging to a given split (train/val/test)."""
        uids = set()
        with open(splits_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    uids.add(row['study_uid'])
        return uids

    def _load_jsonl(self, jsonl_path):
        """Load subject sentences from JSONL file."""
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('valid_json', False) and len(data.get('extracted_sentences', [])) > 0:
                        uid = data['volume_name']
                        if self.split_uids is not None and uid not in self.split_uids:
                            continue
                        mapping[uid] = data['extracted_sentences']
                except Exception:
                    continue
        return mapping

    def _load_labels(self, labels_file):
        """Load labels CSV: first column = study_uid, rest = binary label columns."""
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            id_col = 'study_uid' if 'study_uid' in reader.fieldnames else 'subject_id'
            self.label_columns = [c for c in reader.fieldnames if c not in (id_col,)]
            for row in reader:
                sid = row[id_col]
                self.subject_to_labels[sid] = np.array(
                    [float(row[c]) for c in self.label_columns], dtype=np.float32
                )

    def _prepare_samples(self, data_folder):
        """Scan data_folder for NIfTI files.

        Supports two directory layouts (same auto-detection as training):
          1) data_folder/<study_uid>/<space>/img/*.nii.gz
          2) data_folder/batchXX/<study_uid>/img/*.nii.gz
        """
        samples = []

        first_level_dirs = sorted([
            d for d in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, d))
        ])
        if not first_level_dirs:
            return samples

        first_dir = os.path.join(data_folder, first_level_dirs[0])
        use_space_layout = os.path.isdir(os.path.join(first_dir, self.space))

        if use_space_layout:
            for study_uid in first_level_dirs:
                img_dir = os.path.join(data_folder, study_uid, self.space, 'img')
                self._add_subject(samples, study_uid, img_dir)
        else:
            for batch_dir in first_level_dirs:
                batch_path = os.path.join(data_folder, batch_dir)
                for study_uid in sorted(os.listdir(batch_path)):
                    img_dir = os.path.join(batch_path, study_uid, 'img')
                    self._add_subject(samples, study_uid, img_dir)

        return samples

    def _add_subject(self, samples, study_uid, img_dir):
        """Add a subject if it has matching reports and NIfTI files."""
        if not os.path.isdir(img_dir):
            return
        if study_uid not in self.subject_to_sentences:
            return

        nii_files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.nii.gz')
        ])

        if len(nii_files) == 0:
            return

        sample = {
            'subject_id': study_uid,
            'image_paths': nii_files,
            'sentences': self.subject_to_sentences[study_uid],
        }

        if study_uid in self.subject_to_labels:
            sample['labels'] = self.subject_to_labels[study_uid]

        samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def load_and_resample_nii(self, path):
        """Load NIfTI, reorient to RAS, resample to target spacing."""
        nii_img = nib.load(str(path))
        nii_img = nib.as_closest_canonical(nii_img)

        img_data = nii_img.get_fdata().astype(np.float32)
        np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        voxel_sizes = nii_img.header.get_zooms()
        if len(voxel_sizes) >= 3:
            current_spacing = (float(voxel_sizes[2]), float(voxel_sizes[0]), float(voxel_sizes[1]))
        else:
            current_spacing = (1.0, 1.0, 1.0)

        img_data = img_data.transpose(2, 0, 1)  # (X, Y, Z) -> (Z, X, Y)
        tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
        resampled = resize_array(tensor, current_spacing, self.target_spacing)[0, 0]

        return resampled

    def normalize_volume(self, data):
        return self.normalizer_obj.normalize(data)

    def crop_or_pad(self, data):
        """Center crop or pad to target_shape (D, H, W), with posterior shift on W.

        Identical to MRReportDataset.crop_or_pad to keep inference inputs in
        the same anatomical FOV the model was trained on.
        """
        tensor = torch.from_numpy(data.astype(np.float32))

        td, th, tw = self.target_shape
        d, h, w = tensor.shape

        d_start = max((d - td) // 2, 0)
        h_start = max((h - th) // 2, 0)

        w_center = w // 2 - self.posterior_shift_voxels
        w_start = w_center - tw // 2
        w_start = max(w_start, 0)
        w_start = min(w_start, max(w - tw, 0))

        tensor = tensor[d_start:d_start + td, h_start:h_start + th, w_start:w_start + tw]

        pad_d_before = (td - tensor.size(0)) // 2
        pad_d_after = td - tensor.size(0) - pad_d_before
        pad_h_before = (th - tensor.size(1)) // 2
        pad_h_after = th - tensor.size(1) - pad_h_before
        pad_w_before = (tw - tensor.size(2)) // 2
        pad_w_after = tw - tensor.size(2) - pad_w_before

        tensor = F.pad(
            tensor,
            (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
            value=0,
        )

        return tensor.unsqueeze(0).to(torch.bfloat16)  # [1, D, H, W]

    def __getitem__(self, index):
        sample = self.samples[index]

        volume_tensors = []
        for path in sample['image_paths']:
            resampled = self.load_and_resample_nii(path)
            normalized = self.normalize_volume(resampled)
            tensor = self.crop_or_pad(normalized)
            volume_tensors.append(tensor)

        volume_stack = torch.stack(volume_tensors, dim=0)  # [N, 1, D, H, W]
        real_volume_mask = torch.ones(volume_stack.shape[0], dtype=torch.bool)

        sentences = sample['sentences']
        subject_id = sample['subject_id']
        labels = sample.get('labels', np.array([], dtype=np.float32))

        return volume_stack, sentences, subject_id, real_volume_mask, labels


def collate_fn_infer(batch):
    """Collate for batch_size=1 inference. Unwrap the single item."""
    images, sentences, subject_id, mask, labels = batch[0]
    return (
        images.unsqueeze(0),       # [1, N, 1, D, H, W]
        sentences,                  # list of str
        subject_id,                 # str
        mask.unsqueeze(0),          # [1, N]
        labels,                     # np.ndarray
    )
