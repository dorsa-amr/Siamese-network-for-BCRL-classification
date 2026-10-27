"""Evaluation of a trained method (paper Sec. IV, Table IV and Supplementary Tables I and III).

    python evaluate.py checkpoints/multimodal [--oof-dir outputs/oof/multimodal]

The experiment folder must contain Fold1, Fold2 and Fold3 sub-folders with the checkpoints of the five runs of
each fold, as saved by the training scripts. Two strategies are reported:
    Averaged: metrics of each run at its best validation loss, averaged over the five runs of a fold.
    Ensemble: validation probabilities of the five runs averaged per sample, then thresholded at 0.5.
Fold-level values are summarized as mean +- SD and 95% confidence interval over the three folds (t-distribution).
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, confusion_matrix

from common import compute_metrics

METRICS = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']


def specificity(labels, preds):
    tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def load_runs(fold_dir):
    paths = sorted(glob.glob(os.path.join(fold_dir, '*.pth')))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {fold_dir}")
    runs = []
    for path in paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        runs.append({
            'path': path,
            'epoch': ckpt['best_val_epoch'],
            'probs': np.asarray(ckpt['bestepoch_probs']).ravel(),
            'labels': np.asarray(ckpt['bestepochs_labels']).ravel().astype(int),
            'names': list(ckpt['bestepoch_img_names']),
        })
    return runs


def averaged_metrics(runs):
    """Mean over runs of the metrics of each run (stored probabilities thresholded at 0.5)."""
    per_run = []
    for run in runs:
        m = compute_metrics(run['labels'], run['probs'])
        m['specificity'] = specificity(run['labels'], (run['probs'] > 0.5).astype(int))
        per_run.append(m)
    return {k: np.mean([m[k] for m in per_run]) for k in METRICS}


def ensemble_predictions(runs):
    """Averages the probabilities of the same validation sample over all runs."""
    probs, labels = defaultdict(list), {}
    for run in runs:
        assert len(run['names']) == len(run['probs']) == len(run['labels']), f"Length mismatch in {run['path']}"
        for name, prob, label in zip(run['names'], run['probs'], run['labels']):
            probs[name].append(prob)
            labels[name] = label
    names = sorted(probs)
    return names, np.array([labels[n] for n in names]), np.array([np.mean(probs[n]) for n in names])


def ensemble_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    m = compute_metrics(labels, probs, preds)
    m['specificity'] = specificity(labels, preds)
    m['auprc'] = average_precision_score(labels, probs)
    return m


def summarize(values):
    """Mean, SD and 95% CI over folds (in %)."""
    values = 100 * np.asarray(values)
    mean, sd = values.mean(), values.std(ddof=1)
    half = stats.t.ppf(0.975, df=len(values) - 1) * sd / np.sqrt(len(values))
    return f"{mean:5.1f} +- {sd:4.1f}  [{mean - half:5.1f}, {mean + half:5.1f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment_dir')
    parser.add_argument('--oof-dir', help='save the ensemble (out-of-fold) predictions of each fold as CSV files here')
    args = parser.parse_args()

    fold_dirs = [os.path.join(args.experiment_dir, f'Fold{fold}') for fold in (1, 2, 3)]
    averaged, ensembled = [], []
    for fold_dir in fold_dirs:
        fold = os.path.basename(fold_dir)
        runs = load_runs(fold_dir)
        avg = averaged_metrics(runs)
        names, labels, probs = ensemble_predictions(runs)
        ens = ensemble_metrics(labels, probs)
        averaged.append(avg)
        ensembled.append(ens)

        print(f"\n{fold}: {len(runs)} runs, {len(names)} validation samples, best epochs {[r['epoch'] for r in runs]}")
        print("  Averaged: " + ", ".join(f"{k} {100 * avg[k]:.1f}" for k in METRICS))
        print("  Ensemble: " + ", ".join(f"{k} {100 * ens[k]:.1f}" for k in METRICS + ['auprc']))

        if args.oof_dir:
            os.makedirs(args.oof_dir, exist_ok=True)
            out = os.path.join(args.oof_dir, f'{fold}.csv')
            pd.DataFrame({'sample': names, 'y_true': labels, 'p_pred': probs}).to_csv(out, index=False)
            print(f"  Saved {out}")

    print(f"\nMean +- SD [95% CI] over {len(fold_dirs)} folds (%)")
    for title, results, keys in [('Averaged', averaged, METRICS), ('Ensemble', ensembled, METRICS + ['auprc'])]:
        print(title)
        for k in keys:
            print(f"  {k:<12}{summarize([r[k] for r in results])}")


if __name__ == "__main__":
    main()
