"""Two-dimensional t-SNE of the concatenated image and tabular embeddings of the Multimodal model (paper Fig. 4).

For each fold, the validation samples of that fold are embedded with one trained model of the same fold
(by default the run with the lowest validation loss), and the embeddings of the three folds are combined.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from common import DATA_DIR, DEVICE, OUT_DIR, SAVE_DIR, TABULAR_FILE, VAL_FOLDS, transform
from dataset import LymphedemaPairDataset_3frames
from model import Multimodal

MODELS_DIR = os.path.join(SAVE_DIR, 'multimodal')  # contains Fold1, Fold2, Fold3
PERPLEXITY = 10


def best_checkpoint(fold_dir):
    paths = sorted(glob.glob(os.path.join(fold_dir, '*.pth')))
    losses = [torch.load(p, map_location='cpu', weights_only=False)['best_val_loss'] for p in paths]
    return paths[int(np.argmin(losses))]


def extract_features(model, loader):
    features, labels = [], []
    with torch.no_grad():
        for image1, image2, x_tab, label, *_ in loader:
            image1, image2, x_tab = image1.to(DEVICE), image2.to(DEVICE), x_tab.to(DEVICE)
            features.append(model.combined_embedding(image1, image2, x_tab).cpu())
            labels.append(label)
    return torch.cat(features).numpy(), torch.cat(labels).numpy().astype(int)


if __name__ == "__main__":
    model = Multimodal().to(DEVICE).eval()
    all_features, all_labels = [], []
    for fold_idx, val_ids in enumerate(VAL_FOLDS):
        ckpt_path = best_checkpoint(os.path.join(MODELS_DIR, f'Fold{fold_idx + 1}'))
        print(f"Fold {fold_idx + 1}: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state_dict'])

        val_dataset = LymphedemaPairDataset_3frames(DATA_DIR, val_ids, TABULAR_FILE, transform, return_tabular=True)
        features, labels = extract_features(model, DataLoader(val_dataset, batch_size=10, shuffle=False))
        all_features.append(features)
        all_labels.append(labels)

    features, labels = np.concatenate(all_features), np.concatenate(all_labels)
    print("Feature shape:", features.shape)

    features_2d = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42).fit_transform(features)
    label_names = np.where(labels == 1, "1 (Lymphedema)", "0 (No Lymphedema)")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=label_names, palette="deep", legend="full")
    plt.xlabel("t-SNE Dim 1", fontsize=18)
    plt.ylabel("t-SNE Dim 2", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(title="Class", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"tsne_perplexity_{PERPLEXITY}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
