import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm


REBALANCE_STRATEGIES = ('inverse_freq', 'sqrt_inverse_freq', 'max_inverse_freq')


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


# Mapping from logical space name to the image subdirectory used in the
# raw HuggingFace download layout (layout 2). The native repo stores volumes
# in `img/`, while derivative repos (coreg, atlas) use prefixed names.
SPACE_TO_IMG_SUBDIR = {
    'native_space': 'img',
    'coreg_space': 'coreg_img',
    'atlas_space': 'atlas_img',
}

# Filename of the JSON manifest written next to a preprocessed cache. Records the
# exact preprocessing config so the dataloader can refuse to silently train on a
# cache built with a different spacing / shape / normalizer.
CACHE_MANIFEST_NAME = "_manifest.json"


def list_nii_files(img_dir):
    """Sorted list of absolute *.nii.gz paths in img_dir (empty if it's missing)."""
    if not os.path.isdir(img_dir):
        return []
    return sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith('.nii.gz')
    )


def discover_subjects(data_folder, space):
    """Discover subjects and their NIfTI volumes under data_folder for a space.

    Report/split agnostic: returns every subject directory that has >=1 NIfTI in
    its image subdir. Shared by the live dataset (data.py) and the offline
    preprocessing script (preprocess_volumes.py) so discovery never drifts.

    Supports the two layouts documented in MRReportDataset._prepare_samples:
      1) data_folder/<study_uid>/<space>/img/*.nii.gz
      2) data_folder/batchXX/<study_uid>/<img_subdir>/*.nii.gz

    Returns a list of {'subject_id': str, 'image_paths': [str, ...]}.
    """
    first_level_dirs = sorted([
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ])
    if not first_level_dirs:
        return []

    # Auto-detect layout: layout 1 if the first entry has a <space> subfolder.
    first_dir = os.path.join(data_folder, first_level_dirs[0])
    use_space_layout = os.path.isdir(os.path.join(first_dir, space))

    found = []
    if use_space_layout:
        for study_uid in first_level_dirs:
            img_dir = os.path.join(data_folder, study_uid, space, 'img')
            nii = list_nii_files(img_dir)
            if nii:
                found.append({'subject_id': study_uid, 'image_paths': nii})
    else:
        img_subdir = SPACE_TO_IMG_SUBDIR.get(space, 'img')
        for batch_dir in first_level_dirs:
            batch_path = os.path.join(data_folder, batch_dir)
            for study_uid in sorted(os.listdir(batch_path)):
                img_dir = os.path.join(batch_path, study_uid, img_subdir)
                nii = list_nii_files(img_dir)
                if nii:
                    found.append({'subject_id': study_uid, 'image_paths': nii})
    return found


def load_and_resample_nii(path, target_spacing):
    """Load NIfTI, reorient to canonical RAS, resample to target spacing.

    Returns a float32 numpy array of shape (D, H, W) = (Z, X, Y) in RAS.
    """
    nii_img = nib.load(str(path))
    # Reorient to canonical RAS so axes are always (R, A, S)
    nii_img = nib.as_closest_canonical(nii_img)

    img_data = nii_img.get_fdata().astype(np.float32)
    np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    voxel_sizes = nii_img.header.get_zooms()
    if len(voxel_sizes) >= 3:
        # RAS: dim0=R(X), dim1=A(Y), dim2=S(Z). Reorder spacing to (Z, X, Y).
        current_spacing = (float(voxel_sizes[2]), float(voxel_sizes[0]), float(voxel_sizes[1]))
    else:
        current_spacing = (1.0, 1.0, 1.0)

    img_data = img_data.transpose(2, 0, 1)  # (X, Y, Z) -> (Z, X, Y)
    tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
    resampled = resize_array(tensor, current_spacing, target_spacing)[0, 0]
    return resampled


def crop_or_pad(data, target_shape, posterior_shift_voxels):
    """Center crop/pad a (D, H, W) array to target_shape, with posterior W shift.

    The W axis (Y in RAS = anterior-posterior) is shifted posteriorly by
    posterior_shift_voxels to compensate for defacing; if the shift pushes past
    the posterior edge the crop starts at index 0. Pad value is 0 (background).

    Returns a float32 numpy array of shape target_shape.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))

    td, th, tw = target_shape
    d, h, w = tensor.shape

    # Center crop start indices
    d_start = max((d - td) // 2, 0)
    h_start = max((h - th) // 2, 0)

    # W axis (Y/AP): shift center posteriorly (toward lower index in RAS)
    w_center = w // 2 - posterior_shift_voxels
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
    return tensor.numpy()


def preprocess_nii(path, target_spacing, target_shape, posterior_shift_voxels, normalizer_obj):
    """Full single-volume transform: load+RAS+resample -> normalize -> crop/pad.

    Returns a float32 numpy array of shape target_shape (D, H, W). This is the
    single source of truth for the per-volume preprocessing, shared by the live
    dataset and the offline preprocessing script so train/cache stay identical.
    """
    resampled = load_and_resample_nii(path, target_spacing)
    normalized = normalizer_obj.normalize(resampled)
    return crop_or_pad(normalized, target_shape, posterior_shift_voxels)


def build_cache_manifest(space, target_spacing, target_shape, posterior_shift_mm,
                         normalizer, normalizer_kwargs, dtype):
    """Build the dict written as the cache manifest / checked by the dataloader."""
    return {
        'version': 1,
        'layout': 'per_subject_stack',  # one .npz per subject: volumes [N, D, H, W]
        'space': space,
        'target_spacing': list(target_spacing),
        'target_shape': list(target_shape),
        'posterior_shift_mm': float(posterior_shift_mm),
        'normalizer': normalizer,
        'normalizer_kwargs': normalizer_kwargs or {},
        'dtype': dtype,
    }


# Keys whose mismatch between a cache manifest and a requested config means the
# cached volumes are not interchangeable with live preprocessing.
CACHE_CONFIG_KEYS = (
    'space', 'target_spacing', 'target_shape',
    'posterior_shift_mm', 'normalizer', 'normalizer_kwargs',
)


def validate_cache_manifest(space_dir, space, target_spacing, target_shape,
                            posterior_shift_mm, normalizer, normalizer_kwargs,
                            allow_mismatch=False, tag="MRReportDataset"):
    """Check a cache's _manifest.json matches the requested preprocessing config.

    Mismatched spacing / shape / posterior shift / normalizer would make the
    cached volumes silently inconsistent with the live pipeline, so by default
    this raises. Set allow_mismatch=True to downgrade to a warning. Shared by the
    training and inference datasets so the contract is enforced identically.
    """
    manifest_path = os.path.join(space_dir, CACHE_MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        msg = (f"No cache manifest at {manifest_path}; cannot verify the "
               f"preprocessing config matches.")
        if allow_mismatch:
            print(f"[{tag}] WARNING: {msg}")
            return
        raise FileNotFoundError(msg + " Pass cache_allow_mismatch=True to skip this check.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    requested = build_cache_manifest(
        space, target_spacing, target_shape, posterior_shift_mm,
        normalizer, normalizer_kwargs, manifest.get('dtype', 'float16'),
    )
    diffs = []
    for key in CACHE_CONFIG_KEYS:
        want = requested[key]
        got = manifest.get(key)
        if isinstance(want, list) and got is not None:
            got = list(got)
        if got != want:
            diffs.append(f"    {key}: cache={got!r} != requested={want!r}")
    if diffs:
        msg = ("Preprocessed cache config does not match requested config:\n"
               + "\n".join(diffs))
        if allow_mismatch:
            print(f"[{tag}] WARNING: {msg}")
        else:
            raise ValueError(
                msg + "\n  Re-run preprocess_volumes.py with matching args, point "
                "--preprocessed_dir elsewhere, or pass cache_allow_mismatch=True."
            )


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
        target_spacing=(1.0, 0.5, 0.5),
        target_shape=(256, 384, 384),
        posterior_shift_mm=15.0,
        space="native_space",
        normalizer="zscore",
        normalizer_kwargs=None,
        splits_csv=None,
        split="train",
        pathology_labels_csv=None,
        rebalance_strategy=None,
        rebalance_base_weight=1.0,
        rebalance_eps=1e-6,
        preprocessed_dir=None,
        use_preprocessed=False,
        cache_allow_mismatch=False,
    ):
        self.data_folder = data_folder
        self.space = space
        self.max_sentences = max_sentences_per_image
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.posterior_shift_mm = posterior_shift_mm
        # Posterior shift in voxels on Y axis (W dim) to compensate for defacing
        self.posterior_shift_voxels = int(round(posterior_shift_mm / target_spacing[2]))

        # Preprocessed (.npz) cache settings. When use_preprocessed is True the
        # expensive NIfTI read + RAS reorient + resample + normalize + crop is
        # skipped at train time and volumes are read straight from .npz produced
        # by preprocess_volumes.py.
        self.preprocessed_dir = preprocessed_dir
        self.use_preprocessed = bool(use_preprocessed)
        self.cache_allow_mismatch = cache_allow_mismatch

        # Initialize normalizer (also recorded for cache-manifest validation)
        if normalizer not in NORMALIZERS:
            raise ValueError(f"Unknown normalizer '{normalizer}'. Choose from: {list(NORMALIZERS.keys())}")
        self.normalizer_name = normalizer
        self.normalizer_kwargs = normalizer_kwargs or {}
        self.normalizer_obj = NORMALIZERS[normalizer](**self.normalizer_kwargs)

        # Load split filter
        self.split_uids = self._load_splits(splits_csv, split) if splits_csv else None

        # Load reports
        self.subject_to_sentences = self._load_jsonl(jsonl_file)

        # Discover subjects — from the .npz cache or the raw NIfTI tree.
        if self.use_preprocessed:
            if not self.preprocessed_dir:
                raise ValueError("use_preprocessed=True requires preprocessed_dir.")
            self.samples = self._prepare_samples_from_cache()
        else:
            if not data_folder:
                raise ValueError("data_folder is required when use_preprocessed=False.")
            self.samples = self._prepare_samples(data_folder)

        # Optional inverse-prevalence rebalancing weights for rare pathologies
        self.rebalance_strategy = rebalance_strategy
        self.label_columns = []
        self.label_prevalence = None
        self.sample_weights = self._compute_sample_weights(
            pathology_labels_csv,
            rebalance_strategy,
            rebalance_base_weight,
            rebalance_eps,
        )

        src = "preprocessed .npz" if self.use_preprocessed else "raw NIfTI"
        print(f"[MRReportDataset] Found {len(self.samples)} subjects (source: {src})")
        for s in self.samples[:5]:
            n_vols = len(s['image_paths']) if 'image_paths' in s else '?'
            print(f"  - {s['subject_id']}: {n_vols} volumes, {len(s['sentences'])} sentences")
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
        """Scan data_folder for NIfTI files, keeping subjects that have reports.

        Discovery (layout auto-detection + NIfTI listing) is delegated to the
        shared discover_subjects() helper; here we only filter to subjects that
        have matching reports in the (split-filtered) JSONL.
        """
        samples = []
        for sub in discover_subjects(data_folder, self.space):
            sid = sub['subject_id']
            if sid not in self.subject_to_sentences:
                continue
            samples.append({
                'subject_id': sid,
                'image_paths': sub['image_paths'],
                'sentences': self.subject_to_sentences[sid],
            })
        return samples

    def _cache_space_dir(self):
        """Directory holding this space's .npz files: <preprocessed_dir>/<space>/."""
        return os.path.join(self.preprocessed_dir, self.space)

    def _check_cache_manifest(self, space_dir):
        """Verify the cache was built with this dataset's preprocessing config."""
        validate_cache_manifest(
            space_dir, self.space, self.target_spacing, self.target_shape,
            self.posterior_shift_mm, self.normalizer_name, self.normalizer_kwargs,
            allow_mismatch=self.cache_allow_mismatch, tag="MRReportDataset",
        )

    def _prepare_samples_from_cache(self):
        """List preprocessed .npz files, keeping subjects that have reports."""
        space_dir = self._cache_space_dir()
        if not os.path.isdir(space_dir):
            raise FileNotFoundError(
                f"Preprocessed cache dir not found: {space_dir}. Run "
                f"preprocess_volumes.py --out_dir {self.preprocessed_dir} "
                f"--space {self.space} first."
            )
        self._check_cache_manifest(space_dir)

        samples = []
        for fn in sorted(os.listdir(space_dir)):
            if not fn.endswith('.npz'):
                continue
            sid = fn[:-len('.npz')]
            if sid not in self.subject_to_sentences:
                continue
            samples.append({
                'subject_id': sid,
                'cache_path': os.path.join(space_dir, fn),
                'sentences': self.subject_to_sentences[sid],
            })
        return samples

    def _compute_sample_weights(self, csv_path, strategy, base_weight, eps):
        """Compute per-subject sampling weights from a pathology-labels CSV.

        Inverse-prevalence weighting upsamples subjects with rare positive
        pathologies so contrastive batches see them more often. Subjects not
        listed in the CSV (or all-negative subjects) receive `base_weight`.

        Strategies:
          - 'inverse_freq':       base + sum_p y_p * (1 / prevalence_p)
          - 'sqrt_inverse_freq':  base + sum_p y_p * sqrt(1 / prevalence_p)
          - 'max_inverse_freq':   max(base, max_p y_p * (1 / prevalence_p))

        Returns:
            A torch.FloatTensor of length len(self.samples) if rebalancing is
            enabled, else None. Weights are unnormalized (WeightedRandomSampler
            normalizes internally).
        """
        if csv_path is None or strategy is None:
            return None
        if strategy not in REBALANCE_STRATEGIES:
            raise ValueError(
                f"Unknown rebalance_strategy '{strategy}'. "
                f"Choose from: {list(REBALANCE_STRATEGIES)}"
            )

        labels_by_uid, label_columns = self._load_pathology_labels(csv_path)
        self.label_columns = label_columns

        # Compute prevalence over the subset of dataset subjects that have labels
        label_rows = [labels_by_uid[s['subject_id']] for s in self.samples
                      if s['subject_id'] in labels_by_uid]
        if not label_rows:
            print(
                f"[MRReportDataset] WARNING: no dataset subjects matched the "
                f"pathology labels CSV; rebalancing disabled."
            )
            return None
        label_matrix = np.stack(label_rows, axis=0)
        prevalence = label_matrix.mean(axis=0)
        self.label_prevalence = prevalence
        inv_freq = 1.0 / np.clip(prevalence, eps, None)

        if strategy == 'sqrt_inverse_freq':
            per_class = np.sqrt(inv_freq)
        else:
            per_class = inv_freq

        weights = np.full(len(self.samples), base_weight, dtype=np.float32)
        for i, s in enumerate(self.samples):
            y = labels_by_uid.get(s['subject_id'])
            if y is None:
                continue
            if strategy == 'max_inverse_freq':
                pos = y * inv_freq
                weights[i] = max(base_weight, float(pos.max()))
            else:
                weights[i] = base_weight + float((y * per_class).sum())

        n_labeled = sum(1 for s in self.samples if s['subject_id'] in labels_by_uid)
        print(
            f"[MRReportDataset] Rebalancing enabled (strategy={strategy}): "
            f"{n_labeled}/{len(self.samples)} subjects matched labels CSV, "
            f"weight range=[{weights.min():.3g}, {weights.max():.3g}], "
            f"mean={weights.mean():.3g}"
        )
        return torch.from_numpy(weights)

    @staticmethod
    def _load_pathology_labels(csv_path):
        """Load a pathology labels CSV.

        Expects a header row with 'study_uid' (or 'subject_id') followed by
        one binary column per pathology. Returns (dict uid -> np.ndarray,
        list of label column names).
        """
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if 'study_uid' in fields:
                id_col = 'study_uid'
            elif 'subject_id' in fields:
                id_col = 'subject_id'
            else:
                raise ValueError(
                    f"Pathology labels CSV {csv_path} must have a 'study_uid' "
                    f"or 'subject_id' column. Got: {fields}"
                )
            label_columns = [c for c in fields if c != id_col]
            for row in reader:
                labels[row[id_col]] = np.array(
                    [float(row[c]) for c in label_columns], dtype=np.float32
                )
        return labels, label_columns

    def get_weighted_sampler(self, num_samples=None, generator=None):
        """Build a WeightedRandomSampler from the precomputed sample weights.

        Replacement sampling is required so high-weight (rare-pathology)
        subjects can be drawn multiple times per epoch.
        """
        if self.sample_weights is None:
            raise RuntimeError(
                "sample_weights not computed; pass pathology_labels_csv and "
                "rebalance_strategy to the dataset constructor."
            )
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=num_samples or len(self.samples),
            replacement=True,
            generator=generator,
        )

    def __len__(self):
        return len(self.samples)

    def load_and_resample_nii(self, path):
        """Load NIfTI, reorient to RAS, resample to target spacing (np [D,H,W])."""
        return load_and_resample_nii(path, self.target_spacing)

    def normalize_volume(self, data):
        """Normalize volume using the configured normalizer."""
        return self.normalizer_obj.normalize(data)

    def crop_or_pad(self, data):
        """Center crop/pad to target_shape with posterior W shift -> [1,D,H,W] bf16."""
        arr = crop_or_pad(data, self.target_shape, self.posterior_shift_voxels)
        return torch.from_numpy(arr).unsqueeze(0).to(torch.bfloat16)  # [1, D, H, W]

    def _load_volume_stack(self, sample):
        """Return the [N, 1, D, H, W] bf16 volume stack for a subject.

        Reads preprocessed .npz when caching is enabled, otherwise runs the live
        load -> normalize -> crop pipeline on each NIfTI.
        """
        if self.use_preprocessed:
            cached = np.load(sample['cache_path'])
            vols = cached['volumes']  # [N, D, H, W]
            # float16 -> bf16, add channel dim -> [N, 1, D, H, W]
            stack = torch.from_numpy(np.ascontiguousarray(vols)).to(torch.bfloat16)
            return stack.unsqueeze(1)

        volume_tensors = []
        for vi, path in enumerate(sample['image_paths']):
            resampled = self.load_and_resample_nii(path)
            normalized = self.normalize_volume(resampled)
            tensor = self.crop_or_pad(normalized)  # [1, D, H, W]
            volume_tensors.append(tensor)
            if vi == 0:
                print(f"[Dataset]   vol 0 loaded: {tensor.shape}", flush=True)
        return torch.stack(volume_tensors, dim=0)  # [N, 1, D, H, W]

    def __getitem__(self, index):
        sample = self.samples[index]
        all_sentences = sample['sentences']

        print(f"[Dataset] Loading subject {sample['subject_id']}...", flush=True)

        volume_stack = self._load_volume_stack(sample)  # [N, 1, D, H, W]
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
