"""
NeuroVFM Classify-Then-Aggregate MIL on frozen MR-RATE token features.

The head matches NeuroVFM's diagnostic architecture while replacing only
its visual encoder output with projected tokens from a contrastively
pretrained MR-RATE checkpoint:

  attention = W(tanh(V(x)) * sigmoid(U(x)))
  patch_logits = MLP(x)
  bag_logit[c] = sum_i softmax_i(attention[i, c]) * patch_logits[i, c]

The encoder is frozen by construction: extract_features.py runs it once
under no_grad/eval and this script trains only the MIL head. Standard
nn.Linear is used instead of NeuroVFM's FlashAttention FusedDense; the
parameterization and computation are otherwise the same.

Workflow:
  python scripts/extract_features.py ... --split train \
      --feature_level tokens --cache_dtype float16 --out_dir mil_features
  python scripts/extract_features.py ... --split val \
      --feature_level tokens --cache_dtype float16 --out_dir mil_features
  python scripts/extract_features.py ... --split test \
      --feature_level tokens --cache_dtype float16 --out_dir mil_features
  python scripts/mil_probe.py \
      --features_dir mil_features --results_dir mil_results

Exact token caches can be very large. extract_features.py offers an
explicit --max_tokens_per_study approximation; leaving it at 0 preserves
the complete encoder token bag.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class PatchClassifier(nn.Module):
    """NeuroVFM's shared MLP: Linear/GELU hidden blocks, then Linear."""

    def __init__(self, dim: int, hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        current = dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, hidden), nn.GELU()))
            current = hidden
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifyThenAggregate(nn.Module):
    """NeuroVFM's class-specific gated-attention MIL head.

    Inputs are flat tokens from a batch of ragged bags plus cumulative bag
    boundaries. Softmax is independently normalized within each bag and
    class, exactly matching NeuroVFM's segment_csr implementation.
    """

    def __init__(
        self,
        dim: int,
        n_classes: int,
        hidden_dim: int = 512,
        mlp_hidden_dims: tuple[int, ...] = (384,),
        drop_rate: float = 0.0,
        init_std: float = 0.02,
        use_gating: bool = True,
        use_norm: bool = False,
        use_output_bias_scale: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.use_gating = use_gating
        self.use_output_bias_scale = use_output_bias_scale
        self.init_std = init_std

        norm = lambda: nn.LayerNorm(dim, eps=1e-6) if use_norm else nn.Identity()
        self.attention_norm = norm()
        self.classifier_norm = norm()
        self.input_dropout = nn.Dropout(drop_rate)

        self.attention_v = nn.Linear(dim, hidden_dim)
        self.attention_u = nn.Linear(dim, hidden_dim) if use_gating else None
        self.attention_w = nn.Linear(hidden_dim, n_classes)
        self.patch_classifier = PatchClassifier(dim, mlp_hidden_dims, n_classes)

        if use_output_bias_scale:
            self.output_scale = nn.Parameter(torch.ones(n_classes))
            self.output_bias = nn.Parameter(torch.zeros(n_classes))
        else:
            self.register_parameter("output_scale", None)
            self.register_parameter("output_bias", None)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        return_details: bool = False,
    ):
        if tokens.ndim != 2 or tokens.shape[1] != self.dim:
            raise ValueError(f"tokens must have shape [N, {self.dim}], got {tuple(tokens.shape)}")
        bounds = cu_seqlens.detach().cpu().tolist()
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != tokens.shape[0]:
            raise ValueError("cu_seqlens must start at 0 and end at the flat token count")

        attention_input = self.input_dropout(self.attention_norm(tokens))
        classifier_input = self.input_dropout(self.classifier_norm(tokens))

        attention_hidden = torch.tanh(self.attention_v(attention_input))
        if self.use_gating:
            attention_hidden = attention_hidden * torch.sigmoid(
                self.attention_u(attention_input)
            )
        attention_scores = self.attention_w(attention_hidden)
        patch_logits = self.patch_classifier(classifier_input)

        bag_logits: list[torch.Tensor] = []
        attention_chunks: list[torch.Tensor] = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                raise ValueError("MIL bags must contain at least one token")
            class_attention = F.softmax(attention_scores[start:end], dim=0)
            bag_logits.append((class_attention * patch_logits[start:end]).sum(dim=0))
            if return_details:
                attention_chunks.append(class_attention)

        output = torch.stack(bag_logits, dim=0)
        if self.use_output_bias_scale:
            output = output * self.output_scale + self.output_bias

        if return_details:
            return output, torch.cat(attention_chunks, dim=0), patch_logits
        return output


class RaggedTokenDataset(Dataset):
    """Memory-mapped projected MR-RATE token bags."""

    def __init__(self, features_dir: Path, split: str):
        self.features_dir = features_dir
        self.split = split
        metadata_path = features_dir / f"token_features_{split}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Run extract_features.py with "
                f"--feature_level tokens --split {split}."
            )
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata.get("format") != "raw_numpy_memmap":
            raise ValueError(f"Unsupported token cache format in {metadata_path}")

        self.dim = int(self.metadata["dim"])
        self.dtype = np.dtype(self.metadata["dtype"])
        if self.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError(f"Unsupported token dtype {self.dtype} for split={split}")
        self.offsets = np.load(features_dir / self.metadata["offsets_file"])
        labels_file = self.metadata.get("labels_file", f"labels_{split}.npy")
        ids_file = self.metadata.get("subject_ids_file", f"subject_ids_{split}.txt")
        self.labels = np.load(features_dir / labels_file).astype(np.float32)
        ids_path = features_dir / ids_file
        self.subject_ids = ids_path.read_text().strip().splitlines()

        full_counts_file = self.metadata.get("full_token_counts_file")
        series_counts_file = self.metadata.get("series_counts_file")
        if full_counts_file and series_counts_file:
            self.full_token_counts = np.load(features_dir / full_counts_file)
            self.series_counts = np.load(features_dir / series_counts_file)
        else:
            self.full_token_counts = np.diff(self.offsets).astype(np.int64)
            self.series_counts = np.ones(len(self.labels), dtype=np.int32)

        if self.offsets.ndim != 1 or self.offsets.shape[0] != self.labels.shape[0] + 1:
            raise ValueError(f"Offsets/labels mismatch for split={split}")
        if len(self.labels) == 0:
            raise ValueError(f"Split {split} contains no studies")
        if len(self.subject_ids) != self.labels.shape[0]:
            raise ValueError(f"Study IDs/labels mismatch for split={split}")
        if len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError(f"Duplicate study IDs within split={split}")
        if self.offsets[0] != 0 or np.any(np.diff(self.offsets) <= 0):
            raise ValueError(f"Invalid or empty token bags for split={split}")
        if not np.isfinite(self.labels).all():
            raise ValueError(f"Non-finite labels in split={split}")
        if not np.isin(self.labels, (0.0, 1.0)).all():
            raise ValueError(f"Labels must be strictly binary 0/1 in split={split}")
        if not (
            len(self.full_token_counts) == len(self.series_counts) == len(self.labels)
        ):
            raise ValueError(f"Token mapping metadata mismatch for split={split}")
        if np.any(self.series_counts <= 0) or np.any(self.full_token_counts <= 0):
            raise ValueError(f"Invalid series/token counts for split={split}")
        if np.any(self.full_token_counts % self.series_counts != 0):
            raise ValueError(f"Token counts are not divisible by series counts for split={split}")

        self.num_tokens = int(self.offsets[-1])
        token_path = features_dir / self.metadata["tokens_file"]
        expected_bytes = self.num_tokens * self.dim * self.dtype.itemsize
        if token_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Token cache size mismatch for {token_path}: expected "
                f"{expected_bytes} bytes, found {token_path.stat().st_size}"
            )
        self.tokens = np.memmap(
            token_path,
            mode="r",
            dtype=self.dtype,
            shape=(self.num_tokens, self.dim),
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        # Copy makes the NumPy buffer writable for safe torch collation.
        tokens = torch.from_numpy(np.array(self.tokens[start:end], copy=True))
        labels = torch.from_numpy(self.labels[index])
        return tokens, labels, index

    def token_mapping(self, index: int) -> tuple[np.ndarray, int, int]:
        """Return original flat indices, series count, and tokens per series."""
        cached_count = int(self.offsets[index + 1] - self.offsets[index])
        full_count = int(self.full_token_counts[index])
        series_count = int(self.series_counts[index])
        if cached_count == full_count:
            token_indices = np.arange(full_count, dtype=np.int64)
        else:
            token_indices = np.rint(
                np.linspace(0, full_count - 1, cached_count)
            ).astype(np.int64)
        return token_indices, series_count, full_count // series_count


def collate_ragged(batch):
    token_bags, labels, indices = zip(*batch)
    lengths = torch.tensor([bag.shape[0] for bag in token_bags], dtype=torch.long)
    cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    return (
        torch.cat(token_bags, dim=0),
        torch.stack(labels, dim=0),
        cu_seqlens,
        torch.tensor(indices, dtype=torch.long),
    )


def amp_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def prepare_tokens(tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    tokens = tokens.to(device, non_blocking=True)
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        tokens = tokens.float()
    return tokens


def auroc_table(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import roc_auc_score

    per_class: dict[str, float] = {}
    valid: list[float] = []
    for class_index, name in enumerate(label_names):
        target = labels[:, class_index]
        if len(np.unique(target)) < 2:
            per_class[name] = float("nan")
            continue
        score = float(roc_auc_score(target, logits[:, class_index]))
        per_class[name] = score
        valid.append(score)
    return (float(np.mean(valid)) if valid else float("nan")), per_class


def select_validation_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Select one logit threshold per class using validation labels only."""
    from sklearn.metrics import roc_curve

    thresholds = np.zeros(logits.shape[1], dtype=np.float32)
    for class_index in range(logits.shape[1]):
        target = labels[:, class_index]
        if len(np.unique(target)) < 2:
            continue
        false_positive, true_positive, candidates = roc_curve(
            target, logits[:, class_index]
        )
        objective = true_positive - false_positive
        objective[~np.isfinite(candidates)] = -np.inf
        best = int(np.argmax(objective))
        if np.isfinite(candidates[best]):
            thresholds[class_index] = float(candidates[best])
    return thresholds


def per_class_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    thresholds: np.ndarray,
) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for class_index, name in enumerate(label_names):
        target = labels[:, class_index]
        prediction = logits[:, class_index] >= thresholds[class_index]
        positive = target == 1
        negative = ~positive
        true_positive = int(np.sum(prediction & positive))
        false_negative = int(np.sum(~prediction & positive))
        true_negative = int(np.sum(~prediction & negative))
        false_positive = int(np.sum(prediction & negative))
        has_both = len(np.unique(target)) == 2
        rows.append({
            "label": name,
            "auroc": float(roc_auc_score(target, logits[:, class_index])) if has_both else None,
            "auprc": float(average_precision_score(target, logits[:, class_index])) if positive.any() else None,
            "threshold": float(thresholds[class_index]),
            "sensitivity": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative else None
            ),
            "specificity": (
                true_negative / (true_negative + false_positive)
                if true_negative + false_positive else None
            ),
            "positives": int(positive.sum()),
            "negatives": int(negative.sum()),
        })
    return rows


def evaluate_head(
    head: ClassifyThenAggregate,
    loader: DataLoader,
    device: torch.device,
    attention_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    head.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    indices_out: list[int] = []
    if attention_dir is not None:
        if attention_dir.exists() and any(attention_dir.iterdir()):
            raise RuntimeError(
                f"Attention directory is not empty: {attention_dir}. "
                "Use a new --results_dir to avoid stale artifacts."
            )
        attention_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for tokens, labels, cu_seqlens, indices in loader:
            tokens = prepare_tokens(tokens, device)
            labels = labels.to(device, non_blocking=True)
            with amp_context(device):
                if attention_dir is None:
                    logits = head(tokens, cu_seqlens)
                    attention = patch_logits = None
                else:
                    logits, attention, patch_logits = head(
                        tokens, cu_seqlens, return_details=True
                    )

            logits_out.append(logits.float().cpu().numpy())
            labels_out.append(labels.float().cpu().numpy())
            batch_indices = indices.tolist()
            indices_out.extend(batch_indices)

            if attention_dir is not None:
                bounds = cu_seqlens.tolist()
                dataset = loader.dataset
                for local_index, dataset_index in enumerate(batch_indices):
                    start, end = bounds[local_index], bounds[local_index + 1]
                    subject_id = dataset.subject_ids[dataset_index]
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", subject_id)
                    token_indices, series_count, tokens_per_series = (
                        dataset.token_mapping(dataset_index)
                    )
                    np.savez_compressed(
                        attention_dir / f"{dataset_index:06d}_{safe_id}.npz",
                        attention=attention[start:end].float().cpu().numpy().astype(np.float16),
                        patch_logits=patch_logits[start:end].float().cpu().numpy().astype(np.float16),
                        original_token_indices=token_indices,
                        series_count=np.asarray(series_count, dtype=np.int32),
                        tokens_per_series=np.asarray(tokens_per_series, dtype=np.int32),
                    )

    return np.concatenate(logits_out), np.concatenate(labels_out), indices_out


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
):
    warmup_steps = int(total_steps * warmup_fraction)

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main() -> None:
    parser = argparse.ArgumentParser(
        "NeuroVFM Classify-Then-Aggregate MIL on frozen MR-RATE tokens"
    )
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./mil_probe_results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_fraction", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_hidden_dim", type=int, default=384)
    parser.add_argument("--no_pos_weight", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_test_attention", action="store_true",
                        help="Write per-study class attention and patch logits; this can be large.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch_size must be positive")
    if not 0.0 <= args.warmup_fraction <= 1.0:
        parser.error("--warmup_fraction must be between 0 and 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    label_names: list[str] = json.loads((features_dir / "label_names.json").read_text())
    if not label_names:
        raise ValueError("label_names.json contains no classes")
    train_data = RaggedTokenDataset(features_dir, "train")
    val_data = RaggedTokenDataset(features_dir, "val")
    test_data = RaggedTokenDataset(features_dir, "test")
    fingerprints = {
        dataset.metadata.get("cache_fingerprint")
        for dataset in (train_data, val_data, test_data)
    }
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError(
            "Train/val/test token caches do not share one verified cache fingerprint"
        )
    if not (train_data.dim == val_data.dim == test_data.dim):
        raise ValueError("Feature dimensions differ across splits")
    if not (
        train_data.labels.shape[1]
        == val_data.labels.shape[1]
        == test_data.labels.shape[1]
        == len(label_names)
    ):
        raise ValueError("Label dimensions differ across splits or label_names.json")
    split_ids = {
        "train": set(train_data.subject_ids),
        "val": set(val_data.subject_ids),
        "test": set(test_data.subject_ids),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(
                f"Study leakage between {left} and {right}: {len(overlap)} duplicate IDs"
            )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_ragged, drop_last=False,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_ragged, drop_last=False,
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_ragged, drop_last=False,
    )

    n_classes = len(label_names)
    head = ClassifyThenAggregate(
        dim=train_data.dim,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        mlp_hidden_dims=(args.mlp_hidden_dim,),
        drop_rate=0.0,
        use_gating=True,
        use_norm=False,
        use_output_bias_scale=True,
    ).to(device)
    print(
        f"NeuroVFM CTA head: dim={train_data.dim}, attention_hidden={args.hidden_dim}, "
        f"patch_mlp=[{args.mlp_hidden_dim}], classes={n_classes}"
    )
    print(
        f"Bags: train={len(train_data)}, val={len(val_data)}, test={len(test_data)} | "
        f"tokens(train)={train_data.num_tokens:,}"
    )

    if args.no_pos_weight:
        pos_weight = None
        print("pos_weight: disabled")
    else:
        positives = train_data.labels.sum(axis=0)
        negatives = len(train_data) - positives
        weights = np.clip(negatives / np.clip(positives, 1.0, None), 1.0, 100.0)
        pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"pos_weight: median={float(np.median(weights)):.1f}, max={float(weights.max()):.1f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer, total_steps=max(len(train_loader) * args.epochs, 1),
        warmup_fraction=args.warmup_fraction,
    )

    best_val_auroc = -1.0
    best_epoch = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_loss = 0.0
        n_seen = 0
        for tokens, labels, cu_seqlens, _indices in train_loader:
            tokens = prepare_tokens(tokens, device)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = head(tokens, cu_seqlens)
                loss = loss_fn(logits.float(), labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * labels.shape[0]
            n_seen += labels.shape[0]

        train_loss = epoch_loss / max(n_seen, 1)
        val_logits, val_labels, _ = evaluate_head(head, val_loader, device)
        val_mean, _ = auroc_table(val_logits, val_labels, label_names)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_mean_auroc": val_mean}
        )
        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"val_mean_AUROC={val_mean:.4f}"
        )
        if np.isfinite(val_mean) and val_mean > best_val_auroc:
            best_val_auroc = val_mean
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No epoch produced a finite validation mean AUROC")
    head.load_state_dict(best_state)
    print(f"Best validation mean AUROC: {best_val_auroc:.4f}")

    best_val_logits, best_val_labels, _ = evaluate_head(head, val_loader, device)
    validation_thresholds = select_validation_thresholds(
        best_val_logits, best_val_labels
    )

    attention_dir = results_dir / "test_attention" if args.save_test_attention else None
    test_logits, test_labels, test_indices = evaluate_head(
        head, test_loader, device, attention_dir=attention_dir
    )
    test_mean, test_per_class = auroc_table(test_logits, test_labels, label_names)
    test_metric_rows = per_class_metrics(
        test_logits, test_labels, label_names, validation_thresholds
    )
    print(f"Test mean AUROC: {test_mean:.4f}")

    architecture = {
        "name": "NeuroVFM ClassifyThenAggregate",
        "dim": train_data.dim,
        "n_classes": n_classes,
        "hidden_dim": args.hidden_dim,
        "mlp_hidden_dims": [args.mlp_hidden_dim],
        "drop_rate": 0.0,
        "init_std": 0.02,
        "use_gating": True,
        "use_norm": False,
        "use_output_bias_scale": True,
    }
    torch.save(
        {
            "state_dict": best_state,
            "architecture": architecture,
            "label_names": label_names,
            "cache_provenance": {
                "train": train_data.metadata,
                "val": val_data.metadata,
                "test": test_data.metadata,
            },
            "best_epoch": best_epoch,
            "best_val_mean_auroc": best_val_auroc,
            "validation_thresholds": validation_thresholds,
            "args": vars(args),
        },
        results_dir / "mil_head.pt",
    )
    np.save(results_dir / "test_logits.npy", test_logits.astype(np.float32))
    np.save(results_dir / "test_labels.npy", test_labels.astype(np.float32))
    ordered_test_ids = [test_data.subject_ids[index] for index in test_indices]
    (results_dir / "test_subject_ids.txt").write_text("\n".join(ordered_test_ids) + "\n")
    (results_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (results_dir / "validation_thresholds.json").write_text(
        json.dumps(
            dict(zip(label_names, validation_thresholds.astype(float))), indent=2,
            allow_nan=False,
        ) + "\n"
    )
    (results_dir / "per_class_test_auroc.json").write_text(
        json.dumps(
            {
                "mean_auroc": test_mean,
                "per_class": {
                    name: (value if np.isfinite(value) else None)
                    for name, value in test_per_class.items()
                },
                "metrics": test_metric_rows,
            },
            indent=2,
            allow_nan=False,
        ) + "\n"
    )
    with (results_dir / "test_aurocs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_metric_rows[0]))
        writer.writeheader()
        writer.writerows(test_metric_rows)

    print(f"Artifacts written to {results_dir}")


if __name__ == "__main__":
    main()
