"""
Evaluation utilities for MR-RATE inference.

Computes AUROC, ROC curves, precision-recall curves, and bootstrap CIs.
Adapted from RAD-RATE eval.py for brain MRI pathology classification.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, precision_score,
)
from sklearn.utils import resample


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_roc(y_pred, y_true, roc_name, plot_dir, plot=False):
    """Compute ROC curve and AUROC. Optionally save plot."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    if plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('white')
            fig, ax = plt.subplots(dpi=300)
            ax.set_title(roc_name, fontsize=16)
            ax.plot(fpr, tpr, color='#5C5D9E', linewidth=2, label='AUC = %.2f' % roc_auc)
            ax.fill_between(fpr, tpr, color='#5C5D9E', alpha=0.3)
            ax.plot([0, 1], [0, 1], '--', color='#707071', linewidth=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.legend(loc='lower right')
            ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
            plt.savefig(f"{plot_dir}{roc_name}.png", bbox_inches='tight')
            plt.close()
        except ImportError:
            pass

    return fpr, tpr, thresholds, roc_auc


def choose_operating_point(fpr, tpr, thresholds):
    """Choose threshold that maximizes Youden's J = TPR - FPR."""
    best_j = 0
    best_sens, best_spec = 0, 0
    for _fpr, _tpr in zip(fpr, tpr):
        j = _tpr - _fpr
        if j > best_j:
            best_j = j
            best_sens = _tpr
            best_spec = 1 - _fpr
    return best_sens, best_spec


def plot_pr(y_pred, y_true, pr_name, plot_dir, plot=False):
    """Compute precision-recall curve. Optionally save plot."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    if plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('whitegrid')
            baseline = len(y_true[y_true == 1]) / max(len(y_true), 1)
            fig, ax = plt.subplots(dpi=300)
            ax.set_title(pr_name, fontsize=16)
            ax.plot(recall, precision, color='#5C5D9E', linewidth=2, label='AUC = %.2f' % pr_auc)
            ax.plot([0, 1], [baseline, baseline], '--', color='#707071', linewidth=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.legend(loc='lower right')
            plt.savefig(f"{plot_dir}{pr_name}.png", bbox_inches='tight')
            plt.close()
        except ImportError:
            pass

    return precision, recall, thresholds


def evaluate_internal(y_pred, y_true, labels, plot_dir, plot=False):
    """
    Compute per-class AUROC and PR metrics.

    Args:
        y_pred: (num_samples, num_classes) prediction scores
        y_true: (num_samples, num_classes) binary labels
        labels: list of class names
        plot_dir: directory to save plots (if plot=True)
        plot: whether to generate ROC/PR plots

    Returns:
        DataFrame with AUROC per class
    """
    warnings.filterwarnings('ignore')
    num_classes = y_pred.shape[-1]
    dataframes = []

    for i in range(num_classes):
        y_pred_i = y_pred[:, i]
        y_true_i = y_true[:, i]
        label = labels[i]

        # Skip if only one class present
        if len(np.unique(y_true_i)) < 2:
            print(f"  {label}: skipped (only one class present)")
            df = pd.DataFrame([float('nan')], columns=[label + '_auc'])
            dataframes.append(df)
            continue

        fpr, tpr, thresholds, roc_auc = plot_roc(
            y_pred_i, y_true_i, label + ' ROC', plot_dir, plot=plot
        )
        df = pd.DataFrame([roc_auc], columns=[label + '_auc'])
        dataframes.append(df)

        sens, spec = choose_operating_point(fpr, tpr, thresholds)
        print(f"  {label}: AUROC={roc_auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

        plot_pr(y_pred_i, y_true_i, label + ' PR', plot_dir, plot=plot)

    dfs = pd.concat(dataframes, axis=1)
    return dfs


def find_threshold(probabilities, true_labels):
    """Find optimal threshold that maximizes F1 score."""
    best_threshold = 0.5
    best_f1 = 0

    thresholds = np.unique(probabilities)
    if len(thresholds) > 1000:
        thresholds = np.linspace(0, 1, 101)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        current_f1 = f1_score(true_labels, predictions, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold


def compute_cis(data, confidence_level=0.05):
    """
    Compute confidence intervals from bootstrap samples.

    Args:
        data: DataFrame of shape (num_bootstrap_samples, num_labels)
        confidence_level: significance level (default 0.05 for 95% CI)

    Returns:
        DataFrame with rows [mean, lower, upper]
    """
    intervals = []
    for col in data.columns:
        series = data[col].sort_values()
        lower_idx = int(confidence_level / 2 * len(series)) - 1
        upper_idx = int((1 - confidence_level / 2) * len(series)) - 1
        lower_idx = max(lower_idx, 0)
        upper_idx = min(upper_idx, len(series) - 1)

        lower = round(series.iloc[lower_idx], 4)
        upper = round(series.iloc[upper_idx], 4)
        mean = round(series.mean(), 4)
        intervals.append(pd.DataFrame({col: [mean, lower, upper]}))

    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df


def bootstrap_evaluate(y_pred, y_true, labels, plot_dir,
                       n_samples=100, temperature=14.0):
    """
    Run bootstrap evaluation with sigmoid + temperature scaling.

    Args:
        y_pred: raw prediction scores (num_samples, num_classes)
        y_true: binary labels (num_samples, num_classes)
        labels: list of class names
        plot_dir: directory to save results
        n_samples: number of bootstrap iterations
        temperature: temperature for sigmoid scaling

    Returns:
        dict with DataFrames for auroc, f1, accuracy, precision CIs
    """
    import torch
    from tqdm import tqdm

    # Apply sigmoid with temperature
    pred_tensor = torch.from_numpy(y_pred) / temperature
    pred_scaled = torch.sigmoid(pred_tensor).numpy()

    num_classes = y_pred.shape[-1]

    # Find per-class thresholds
    thresholds = []
    for i in range(num_classes):
        t = find_threshold(pred_scaled[:, i], y_true[:, i])
        thresholds.append(t)
    print(f"Thresholds: {[f'{t:.3f}' for t in thresholds]}")

    # Bootstrap
    auroc_df = pd.DataFrame()
    f1_df = pd.DataFrame()
    acc_df = pd.DataFrame()
    prec_df = pd.DataFrame()

    for _ in tqdm(range(n_samples), desc="Bootstrap"):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        s_labels = y_true[indices]
        s_pred = pred_scaled[indices]

        dfs_auroc = evaluate_internal(s_pred, s_labels, labels, plot_dir, plot=False)
        auroc_df = pd.concat([auroc_df, dfs_auroc])

        f1s, accs, precs = [], [], []
        for i in range(num_classes):
            pred_bin = (s_pred[:, i] > thresholds[i]).astype(int)
            f1s.append(f1_score(s_labels[:, i], pred_bin, zero_division=0))
            accs.append(accuracy_score(s_labels[:, i], pred_bin))
            precs.append(precision_score(s_labels[:, i], pred_bin, zero_division=0))

        f1_df = pd.concat([f1_df, pd.DataFrame([f1s], columns=labels)])
        acc_df = pd.concat([acc_df, pd.DataFrame([accs], columns=labels)])
        prec_df = pd.concat([prec_df, pd.DataFrame([precs], columns=labels)])

    # Save results
    for name, df in [('aurocs', auroc_df), ('f1', f1_df), ('acc', acc_df), ('precision', prec_df)]:
        path = f'{plot_dir}{name}_bootstrap.xlsx'
        df.to_excel(path, index=False, engine='xlsxwriter')

    return {
        'auroc_cis': compute_cis(auroc_df),
        'f1_cis': compute_cis(f1_df),
        'acc_cis': compute_cis(acc_df),
        'precision_cis': compute_cis(prec_df),
    }
