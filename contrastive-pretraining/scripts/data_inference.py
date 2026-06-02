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
from torch.utils.data import Dataset

from data import (
    NORMALIZERS,
    discover_subjects, load_and_resample_nii, crop_or_pad,
    validate_cache_manifest,
)


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
        preprocessed_dir=None,
        use_preprocessed=False,
        cache_allow_mismatch=False,
    ):
        self.data_folder = data_folder
        self.space = space
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.posterior_shift_mm = posterior_shift_mm
        self.posterior_shift_voxels = int(round(posterior_shift_mm / target_spacing[2]))

        # Preprocessed (.npz) cache settings (see MRReportDataset / preprocess_volumes.py)
        self.preprocessed_dir = preprocessed_dir
        self.use_preprocessed = bool(use_preprocessed)
        self.cache_allow_mismatch = cache_allow_mismatch

        if normalizer not in NORMALIZERS:
            raise ValueError(
                f"Unknown normalizer '{normalizer}'. "
                f"Choose from: {list(NORMALIZERS.keys())}"
            )
        self.normalizer_name = normalizer
        self.normalizer_kwargs = normalizer_kwargs or {}
        self.normalizer_obj = NORMALIZERS[normalizer](**self.normalizer_kwargs)

        self.split_uids = self._load_splits(splits_csv, split) if splits_csv else None

        self.subject_to_sentences = self._load_jsonl(jsonl_file)

        self.subject_to_labels = {}
        self.label_columns = []
        if labels_file is not None:
            self._load_labels(labels_file)

        if self.use_preprocessed:
            if not self.preprocessed_dir:
                raise ValueError("use_preprocessed=True requires preprocessed_dir.")
            self.samples = self._prepare_samples_from_cache()
        else:
            if not data_folder:
                raise ValueError("data_folder is required when use_preprocessed=False.")
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
        """Scan data_folder for NIfTI files, keeping subjects with reports.

        Discovery (layout auto-detection + NIfTI listing) is delegated to the
        shared discover_subjects() helper so it never drifts from training.
        """
        samples = []
        for sub in discover_subjects(data_folder, self.space):
            sid = sub['subject_id']
            if sid not in self.subject_to_sentences:
                continue
            sample = {
                'subject_id': sid,
                'image_paths': sub['image_paths'],
                'sentences': self.subject_to_sentences[sid],
            }
            if sid in self.subject_to_labels:
                sample['labels'] = self.subject_to_labels[sid]
            samples.append(sample)
        return samples

    def _prepare_samples_from_cache(self):
        """List preprocessed .npz files, keeping subjects with reports."""
        space_dir = os.path.join(self.preprocessed_dir, self.space)
        if not os.path.isdir(space_dir):
            raise FileNotFoundError(
                f"Preprocessed cache dir not found: {space_dir}. Run "
                f"preprocess_volumes.py --out_dir {self.preprocessed_dir} "
                f"--space {self.space} first."
            )
        validate_cache_manifest(
            space_dir, self.space, self.target_spacing, self.target_shape,
            self.posterior_shift_mm, self.normalizer_name, self.normalizer_kwargs,
            allow_mismatch=self.cache_allow_mismatch, tag="MRReportDatasetInfer",
        )

        samples = []
        for fn in sorted(os.listdir(space_dir)):
            if not fn.endswith('.npz'):
                continue
            sid = fn[:-len('.npz')]
            if sid not in self.subject_to_sentences:
                continue
            sample = {
                'subject_id': sid,
                'cache_path': os.path.join(space_dir, fn),
                'sentences': self.subject_to_sentences[sid],
            }
            if sid in self.subject_to_labels:
                sample['labels'] = self.subject_to_labels[sid]
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def load_and_resample_nii(self, path):
        """Load NIfTI, reorient to RAS, resample to target spacing (np [D,H,W])."""
        return load_and_resample_nii(path, self.target_spacing)

    def normalize_volume(self, data):
        return self.normalizer_obj.normalize(data)

    def crop_or_pad(self, data):
        """Center crop/pad to target_shape with posterior W shift -> [1,D,H,W] bf16."""
        arr = crop_or_pad(data, self.target_shape, self.posterior_shift_voxels)
        return torch.from_numpy(arr).unsqueeze(0).to(torch.bfloat16)  # [1, D, H, W]

    def _load_volume_stack(self, sample):
        """[N,1,D,H,W] bf16 stack — from .npz cache if enabled, else live NIfTI."""
        if self.use_preprocessed:
            cached = np.load(sample['cache_path'])
            vols = cached['volumes']  # [N, D, H, W]
            stack = torch.from_numpy(np.ascontiguousarray(vols)).to(torch.bfloat16)
            return stack.unsqueeze(1)

        volume_tensors = []
        for path in sample['image_paths']:
            resampled = self.load_and_resample_nii(path)
            normalized = self.normalize_volume(resampled)
            tensor = self.crop_or_pad(normalized)
            volume_tensors.append(tensor)
        return torch.stack(volume_tensors, dim=0)  # [N, 1, D, H, W]

    def __getitem__(self, index):
        sample = self.samples[index]

        volume_stack = self._load_volume_stack(sample)  # [N, 1, D, H, W]
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
