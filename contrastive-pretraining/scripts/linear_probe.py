"""
Linear probe on frozen MR-RATE features.

Trains nn.Linear(dim_latent, num_classes) with binary-cross-entropy on
the cached features produced by extract_features.py, evaluates with the
same per-class AUROC pipeline used by inference.py (eval.evaluate_internal),
and dumps predictions, weights, and a metrics table.

Workflow:
  1) python extract_features.py ... --split train --out_dir feats/
     python extract_features.py ... --split val   --out_dir feats/
     python extract_features.py ... --split test  --out_dir feats/
  2) python linear_probe.py --features_dir feats/ --results_dir lp_run/

Why precompute features:
  The 3D encoder is the expensive part. Once it's frozen, every training
  epoch sees the same encoded features, so we run encoding once and then
  train the head in seconds — standard CLIP / SimCLR linear-probe recipe.

Class-imbalance handling:
  Pathologies are very rare (median ~1%). We use BCEWithLogitsLoss with
  per-class positive weights = (#neg / #pos) computed on the training
  split. Pass --no_pos_weight to disable.

Validation:
  Best val mean-AUROC across epochs is checkpointed; final test metrics
  are reported from that checkpoint.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from eval import evaluate_internal


def load_split(features_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feat_path = features_dir / f"features_{split}.npy"
    lab_path = features_dir / f"labels_{split}.npy"
    sid_path = features_dir / f"subject_ids_{split}.txt"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing {feat_path}. Run `extract_features.py --split {split}` first."
        )
    X = np.load(feat_path)
    Y = np.load(lab_path)
    sids = sid_path.read_text().strip().splitlines() if sid_path.exists() else []
    return X.astype(np.float32), Y.astype(np.float32), sids


def evaluate(head: nn.Module, X: torch.Tensor, Y: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    head.eval()
    with torch.no_grad():
        logits = head(X.to(device)).float().cpu().numpy()
    return logits, Y.numpy()


def auroc_table(logits: np.ndarray, Y: np.ndarray, label_names: list[str]) -> tuple[float, dict[str, float]]:
    """Per-class AUROC + macro mean over classes that have both classes present."""
    from sklearn.metrics import roc_auc_score
    per_class = {}
    valid = []
    for j, name in enumerate(label_names):
        yt = Y[:, j]
        if len(np.unique(yt)) < 2:
            per_class[name] = float("nan")
            continue
        a = roc_auc_score(yt, logits[:, j])
        per_class[name] = float(a)
        valid.append(a)
    return (float(np.mean(valid)) if valid else float("nan")), per_class


def main() -> None:
    parser = argparse.ArgumentParser("MR-RATE linear probe on cached features")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing features_<split>.npy / labels_<split>.npy / "
                             "subject_ids_<split>.txt / label_names.json")
    parser.add_argument("--results_dir", type=str, default="./linear_probe_results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_pos_weight", action="store_true",
                        help="Disable per-class positive weighting in BCE loss.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    label_names: list[str] = json.loads((features_dir / "label_names.json").read_text())
    print(f"Labels: {len(label_names)} classes")

    X_tr, Y_tr, _ = load_split(features_dir, "train")
    X_va, Y_va, _ = load_split(features_dir, "val")
    X_te, Y_te, sids_te = load_split(features_dir, "test")

    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1]
    assert Y_tr.shape[1] == Y_va.shape[1] == Y_te.shape[1] == len(label_names), (
        "label-column count mismatches label_names.json — re-run extract_features.py "
        "with the same labels file across splits."
    )
    dim = X_tr.shape[1]
    n_classes = Y_tr.shape[1]
    print(f"  train: {X_tr.shape}, val: {X_va.shape}, test: {X_te.shape}  (dim={dim})")
    print(f"  train positives per class: median={int(np.median(Y_tr.sum(0)))}, "
          f"min={int(Y_tr.sum(0).min())}, max={int(Y_tr.sum(0).max())}")

    # Per-class pos_weight = #neg / #pos to upweight rare positives.
    if args.no_pos_weight:
        pos_weight = None
        print("  pos_weight: disabled")
    else:
        pos = Y_tr.sum(0)
        neg = Y_tr.shape[0] - pos
        # Cap at 100 so a class with 5 positives in 90k doesn't dominate the loss.
        pw = np.clip(neg / np.clip(pos, 1.0, None), 1.0, 100.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=args.device)
        print(f"  pos_weight: median={pw.tolist()[len(pw)//2]:.1f} (capped at 100)")

    # Build the linear head
    head = nn.Linear(dim, n_classes, bias=True).to(args.device)
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, std=0.01)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,
    )
    Xv_t = torch.from_numpy(X_va)
    Yv_t = torch.from_numpy(Y_va)
    Xt_t = torch.from_numpy(X_te)
    Yt_t = torch.from_numpy(Y_te)

    best_val_auroc = -1.0
    best_state = None
    history: list[dict] = []

    print(f"\n--- Training linear head ({args.epochs} epochs, bs={args.batch_size}, lr={args.lr}) ---")
    for epoch in range(1, args.epochs + 1):
        head.train()
        ep_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = ep_loss / max(n_seen, 1)

        v_logits, v_true = evaluate(head, Xv_t, Yv_t, args.device)
        v_mean, _ = auroc_table(v_logits, v_true, label_names)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_auroc": v_mean})
        print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_mean_AUROC={v_mean:.4f}")

        if v_mean > best_val_auroc:
            best_val_auroc = v_mean
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    # Restore best
    assert best_state is not None, "no epoch produced a finite val AUROC"
    head.load_state_dict(best_state)
    print(f"\nBest val mean AUROC: {best_val_auroc:.4f}")

    # Final evaluation
    print("\n--- Test evaluation ---")
    t_logits, t_true = evaluate(head, Xt_t, Yt_t, args.device)
    t_mean, t_per = auroc_table(t_logits, t_true, label_names)
    print(f"Test mean AUROC: {t_mean:.4f}")

    # Save artifacts
    torch.save(
        {"state_dict": head.state_dict(),
         "dim_in": dim, "n_classes": n_classes, "label_names": label_names,
         "args": vars(args)},
        results_dir / "linear_head.pt",
    )
    np.save(results_dir / "test_logits.npy", t_logits.astype(np.float32))
    np.save(results_dir / "test_labels.npy", t_true.astype(np.float32))
    (results_dir / "test_subject_ids.txt").write_text("\n".join(sids_te) + "\n")
    (results_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (results_dir / "per_class_test_auroc.json").write_text(
        json.dumps({"mean_auroc": t_mean, "per_class": t_per}, indent=2) + "\n"
    )

    # Reuse the project's eval pipeline so AUROC numbers are reported identically
    # to inference.py.
    print("\n--- Per-class AUROC (eval.evaluate_internal) ---")
    df = evaluate_internal(t_logits, t_true, label_names, str(results_dir) + "/")
    try:
        df.to_csv(results_dir / "test_aurocs.csv", index=False)
    except Exception as e:
        print(f"Could not save test_aurocs.csv: {e}")

    print(f"\nArtifacts written to {results_dir}")


if __name__ == "__main__":
    main()
