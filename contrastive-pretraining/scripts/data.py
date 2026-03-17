import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm


def cycle(dl):
    """Helper to infinitely loop through a DataLoader."""
    while True:
        for data in dl:
            yield data


def resize_array(array, current_spacing, target_spacing):
    """Resize array to match target spacing using trilinear interpolation."""
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


class ZScoreNormalizer:
    """Z-score on nonzero voxels, clip to [-5,5], rescale to [-1,1]."""

    def normalize(self, data):
        mask = data != 0
        if mask.sum() > 0:
            mean = data[mask].mean()
            std = data[mask].std()
            data = (data - mean) / (std + 1e-8)
        data = np.clip(data, -5.0, 5.0)
        data = data / 5.0
        return data


class PercentileNormalizer:
    """Clip to [lower, upper] percentile, rescale to [lower_limit, upper_limit]."""

    def __init__(self, lower_percentile=0.5, upper_percentile=99.5,
                 lower_limit=-1.0, upper_limit=1.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def normalize(self, data):
        mask = data != 0
        if mask.sum() > 0:
            low = np.percentile(data[mask], self.lower_percentile)
            high = np.percentile(data[mask], self.upper_percentile)
        else:
            low, high = data.min(), data.max()
        data = np.clip(data, low, high)
        if high - low > 1e-8:
            data = (data - low) / (high - low)
            data = data * (self.upper_limit - self.lower_limit) + self.lower_limit
        else:
            data = np.zeros_like(data)
        return data


class MinMaxNormalizer:
    """Simple min-max rescale to [lower_limit, upper_limit]."""

    def __init__(self, lower_limit=-1.0, upper_limit=1.0):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def normalize(self, data):
        dmin = data.min()
        dmax = data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
            data = data * (self.upper_limit - self.lower_limit) + self.lower_limit
        else:
            data = np.zeros_like(data)
        return data


NORMALIZERS = {
    'zscore': ZScoreNormalizer,
    'percentile': PercentileNormalizer,
    'minmax': MinMaxNormalizer,
}


class MRReportDataset(Dataset):
    """
    Dataset for brain MRI with variable numbers of volumes per subject.

    Each subject has a folder with {space}/img/*.nii.gz files (variable count: 2-12+).
    All volumes are loaded, normalized, resampled, and returned as [N, 1, D, H, W]
    where N varies per subject.

    Args:
        space: Which subfolder to load images from ("native_space", "atlas_space", "coreg_space").
        normalizer: Normalization method ("zscore", "percentile", "minmax").
        normalizer_kwargs: Optional kwargs passed to the normalizer constructor.

    With batch_size=1, no padding or masking is needed.
    """

    def __init__(
        self,
        data_folder,
        jsonl_file,
        max_sentences_per_image=34,
        target_spacing=(1.5, 0.75, 0.75),
        target_shape=(256, 480, 480),
        final_spatial_size=(384, 384),
        normalizer="zscore",
        normalizer_kwargs=None,
        splits_csv=None,
        split="train",
    ):
        self.data_folder = data_folder
        self.max_sentences = max_sentences_per_image
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.final_spatial_size = final_spatial_size

        # Initialize normalizer
        if normalizer not in NORMALIZERS:
            raise ValueError(f"Unknown normalizer '{normalizer}'. Choose from: {list(NORMALIZERS.keys())}")
        normalizer_kwargs = normalizer_kwargs or {}
        self.normalizer_obj = NORMALIZERS[normalizer](**normalizer_kwargs)

        # Load split filter
        self.split_uids = self._load_splits(splits_csv, split) if splits_csv else None

        # Load reports
        self.subject_to_sentences = self._load_jsonl(jsonl_file)

        # Discover subjects
        self.samples = self._prepare_samples(data_folder)

        print(f"[MRReportDataset] Found {len(self.samples)} subjects")
        for s in self.samples[:5]:
            print(f"  - {s['subject_id']}: {len(s['image_paths'])} volumes, {len(s['sentences'])} sentences")
        if len(self.samples) > 5:
            print(f"  ... and {len(self.samples) - 5} more")

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

    def _prepare_samples(self, data_folder):
        """Scan data_folder/batchXX/<study_uid>/img/ for NIfTI files."""
        samples = []

        for batch_dir in sorted(os.listdir(data_folder)):
            batch_path = os.path.join(data_folder, batch_dir)
            if not os.path.isdir(batch_path):
                continue

            for study_uid in sorted(os.listdir(batch_path)):
                img_dir = os.path.join(batch_path, study_uid, 'img')
                if not os.path.isdir(img_dir):
                    continue

                if study_uid not in self.subject_to_sentences:
                    continue

                nii_files = sorted([
                    os.path.join(img_dir, f)
                    for f in os.listdir(img_dir)
                    if f.endswith('.nii.gz')
                ])

                if len(nii_files) == 0:
                    continue

                samples.append({
                    'subject_id': study_uid,
                    'image_paths': nii_files,
                    'sentences': self.subject_to_sentences[study_uid],
                })

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

        # Transpose to (D, H, W)
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

        # Center crop if larger
        d_start = max((d - td) // 2, 0)
        h_start = max((h - th) // 2, 0)
        w_start = max((w - tw) // 2, 0)
        tensor = tensor[d_start:d_start + td, h_start:h_start + th, w_start:w_start + tw]

        # Pad if smaller
        pad_d_before = (td - tensor.size(0)) // 2
        pad_d_after = td - tensor.size(0) - pad_d_before
        pad_h_before = (th - tensor.size(1)) // 2
        pad_h_after = th - tensor.size(1) - pad_h_before
        pad_w_before = (tw - tensor.size(2)) // 2
        pad_w_after = tw - tensor.size(2) - pad_w_before

        tensor = F.pad(tensor, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after), value=0)

        # Resize spatial dimensions: [D, H, W] -> [D, final_H, final_W]
        tensor = tensor.unsqueeze(1)  # [D, 1, H, W]
        tensor = F.interpolate(tensor, size=self.final_spatial_size, mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(1).unsqueeze(0)  # [1, D, H, W]

        return tensor.to(torch.bfloat16)

    def __getitem__(self, index):
        sample = self.samples[index]
        all_sentences = sample['sentences']

        print(f"[Dataset] Loading subject {sample['subject_id']} ({len(sample['image_paths'])} volumes)...", flush=True)

        # Load all volumes for this subject
        volume_tensors = []
        for vi, path in enumerate(sample['image_paths']):
            resampled = self.load_and_resample_nii(path)
            normalized = self.normalize_volume(resampled)
            tensor = self.crop_pad_and_resize(normalized)  # [1, D, H, W]
            volume_tensors.append(tensor)
            if vi == 0:
                print(f"[Dataset]   vol 0 loaded: {tensor.shape}", flush=True)

        # Stack: [N, 1, D, H, W] where N = number of volumes
        volume_stack = torch.stack(volume_tensors, dim=0)
        print(f"[Dataset] Subject {sample['subject_id']} done: {volume_stack.shape}", flush=True)

        # Sample/pad sentences
        n = len(all_sentences)
        if n >= self.max_sentences:
            selected = random.sample(all_sentences, self.max_sentences)
            mask = [1] * self.max_sentences
        else:
            padding_count = self.max_sentences - n
            selected = all_sentences + [""] * padding_count
            mask = [1] * n + [0] * padding_count

        return volume_stack, selected, torch.tensor(mask, dtype=torch.bool)


def collate_fn(batch):
    """Collate for batch_size=1. Just unwrap the single item."""
    images, sentences, masks = batch[0]
    # images: [N, 1, D, H, W] - add batch dim -> [1, N, 1, D, H, W]
    return images.unsqueeze(0), sentences, masks.unsqueeze(0)
