"""
Inference dataset for MR-RATE.

Loads brain MRI subjects with variable volumes per subject, supports
configurable space and normalization (reuses normalizers from data.py).

Returns: (images, sentences, subject_id, real_volume_mask)
  - images:            [N, 1, D, H, W]  variable N volumes
  - sentences:         list of report sentences
  - subject_id:        str
  - real_volume_mask:  [N] boolean, all True (no padding at batch_size=1)
"""

import os
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

    Same loading/preprocessing as MRReportDataset but:
    - Deterministic (no random sentence sampling)
    - Returns subject_id for result tracking
    - Returns all sentences (no padding/truncation)
    - Optionally loads labels from a CSV if provided
    """

    def __init__(
        self,
        data_folder,
        jsonl_file,
        target_spacing=(1.5, 0.75, 0.75),
        target_shape=(256, 480, 480),
        final_spatial_size=(384, 384),
        space="native_space",
        normalizer="zscore",
        normalizer_kwargs=None,
        labels_file=None,
    ):
        self.data_folder = data_folder
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.final_spatial_size = final_spatial_size
        self.space = space

        # Initialize normalizer
        if normalizer not in NORMALIZERS:
            raise ValueError(
                f"Unknown normalizer '{normalizer}'. "
                f"Choose from: {list(NORMALIZERS.keys())}"
            )
        normalizer_kwargs = normalizer_kwargs or {}
        self.normalizer_obj = NORMALIZERS[normalizer](**normalizer_kwargs)

        # Load reports
        self.subject_to_sentences = self._load_jsonl(jsonl_file)

        # Load labels if provided
        self.subject_to_labels = {}
        self.label_columns = []
        if labels_file is not None:
            self._load_labels(labels_file)

        # Discover subjects
        self.samples = self._prepare_samples(data_folder)

        print(f"[MRReportDatasetInfer] Found {len(self.samples)} subjects "
              f"(space={space}, normalizer={normalizer})")
        if self.label_columns:
            print(f"[MRReportDatasetInfer] Labels loaded: {len(self.label_columns)} classes")

    def _load_jsonl(self, jsonl_path):
        """Load subject sentences from JSONL file."""
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('valid_json', False) and len(data.get('extracted_sentences', [])) > 0:
                        mapping[data['subject_id']] = data['extracted_sentences']
                except Exception:
                    continue
        return mapping

    def _load_labels(self, labels_file):
        """Load labels CSV: first column = subject_id, rest = binary label columns."""
        import csv
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            self.label_columns = [c for c in reader.fieldnames if c != 'subject_id']
            for row in reader:
                sid = row['subject_id']
                self.subject_to_labels[sid] = np.array(
                    [float(row[c]) for c in self.label_columns], dtype=np.float32
                )

    def _prepare_samples(self, data_folder):
        """Scan data_folder for subject directories with {space}/img/*.nii.gz files."""
        samples = []

        for subject_id in sorted(os.listdir(data_folder)):
            img_dir = os.path.join(data_folder, subject_id, self.space, 'img')
            if not os.path.isdir(img_dir):
                continue

            if subject_id not in self.subject_to_sentences:
                continue

            nii_files = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith('.nii.gz')
            ])

            if len(nii_files) == 0:
                continue

            sample = {
                'subject_id': subject_id,
                'image_paths': nii_files,
                'sentences': self.subject_to_sentences[subject_id],
            }

            if subject_id in self.subject_to_labels:
                sample['labels'] = self.subject_to_labels[subject_id]

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def load_and_resample_nii(self, path):
        """Load NIfTI, resample to target spacing using header info."""
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata().astype(np.float32)
        np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        header = nii_img.header
        voxel_sizes = header.get_zooms()

        if len(voxel_sizes) >= 3:
            current_spacing = (float(voxel_sizes[2]), float(voxel_sizes[0]), float(voxel_sizes[1]))
        else:
            current_spacing = (1.0, 1.0, 1.0)

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
        resampled = resize_array(tensor, current_spacing, self.target_spacing)[0, 0]

        return resampled

    def normalize_volume(self, data):
        """Normalize volume using the configured normalizer."""
        return self.normalizer_obj.normalize(data)

    def crop_pad_and_resize(self, data):
        """Crop/pad to target_shape, then resize spatial dims to final_spatial_size."""
        tensor = torch.from_numpy(data.astype(np.float32))

        td, th, tw = self.target_shape
        d, h, w = tensor.shape

        d_start = max((d - td) // 2, 0)
        h_start = max((h - th) // 2, 0)
        w_start = max((w - tw) // 2, 0)
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

        tensor = tensor.unsqueeze(1)  # [D, 1, H, W]
        tensor = F.interpolate(tensor, size=self.final_spatial_size, mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(1).unsqueeze(0)  # [1, D, H, W]

        return tensor.to(torch.bfloat16)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load all volumes
        volume_tensors = []
        for path in sample['image_paths']:
            resampled = self.load_and_resample_nii(path)
            normalized = self.normalize_volume(resampled)
            tensor = self.crop_pad_and_resize(normalized)
            volume_tensors.append(tensor)

        # [N, 1, D, H, W]
        volume_stack = torch.stack(volume_tensors, dim=0)
        real_volume_mask = torch.ones(volume_stack.shape[0], dtype=torch.bool)

        sentences = sample['sentences']
        subject_id = sample['subject_id']

        # Labels: return empty array if not available
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
        labels,                     # np.array
    )
