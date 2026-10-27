"""Grad-CAM maps of the Unimodal-A model for the validation samples (Supplementary Material, Sec. III).

For each fold, the maps of the five trained runs are averaged (ensemble). Maps are computed on the output of
the last convolutional layer of the ResNet-50 encoder, separately for the left- and right-arm image of a pair.
"""
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DATA_DIR, DEVICE, OUT_DIR, SAVE_DIR, TABULAR_FILE, VAL_FOLDS, transform
from dataset import LymphedemaPairDataset_3frames
from model import UnimodalA

MODELS_DIR = os.path.join(SAVE_DIR, 'unimodal_a')  # contains Fold1, Fold2, Fold3


def load_models(fold_dir):
    models = []
    for path in sorted(glob.glob(os.path.join(fold_dir, '*.pth'))):
        model = UnimodalA()
        model.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'])
        models.append(model.to(DEVICE).eval())
    print(f"Loaded {len(models)} models from {fold_dir}")
    return models


def grad_cam(model, x1, x2):
    """Grad-CAM maps of both images of a pair (normalized to [0, 1]) and the predicted probability."""
    target_layer = model.feature_extractor[7][2].conv3  # last convolutional layer of ResNet-50
    activations = []

    def keep_activation(module, inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = target_layer.register_forward_hook(keep_activation)
    logit = model(x1, x2)  # the hook is called for x1 first, then for x2
    handle.remove()
    model.zero_grad()
    logit.sum().backward()

    cams = []
    for act in activations:
        weights = act.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cams.append((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))
    return cams[0], cams[1], torch.sigmoid(logit).item()


def to_display(x):
    img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img - img.min()) / (img.max() - img.min())


def overlay(img, cam):
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.4 * heatmap + 0.6 * img, 0, 1)


def run_gradcam_ensemble(models, val_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for image1, image2, labels, img1_names, img2_names, *_ in tqdm(val_loader, desc="Grad-CAM"):
        x1, x2 = image1.to(DEVICE), image2.to(DEVICE)
        name1, name2 = img1_names[0].replace('.jpeg', ''), img2_names[0].replace('.jpeg', '')
        label = int(labels.item())

        outputs = [grad_cam(model, x1, x2) for model in models]
        cam1 = np.mean([o[0] for o in outputs], axis=0)
        cam2 = np.mean([o[1] for o in outputs], axis=0)
        avg_prob = np.mean([o[2] for o in outputs])

        # Normalize both maps jointly so that they are comparable
        low, high = min(cam1.min(), cam2.min()), max(cam1.max(), cam2.max())
        cam1, cam2 = (cam1 - low) / (high - low + 1e-8), (cam2 - low) / (high - low + 1e-8)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for ax, x, cam, name in [(axs[0], x1, cam1, name1), (axs[1], x2, cam2, name2)]:
            im = ax.imshow(overlay(to_display(x), cam))
            ax.set_title(name)
            ax.axis('off')
        fig.suptitle(f"Avg Prob: {avg_prob:.2f} | GT: {label}")
        plt.tight_layout()
        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04).set_label('Normalized Activation')
        plt.savefig(os.path.join(save_dir, f"{name1}_{name2}_avgprob{avg_prob:.2f}.png"), bbox_inches='tight')
        plt.close(fig)

        results.append({'img1_name': name1, 'img2_name': name2, 'GT': label, 'avg_prob': avg_prob})

    # Class 0 and class 1 samples side by side, each sorted by the predicted probability
    df = pd.DataFrame(results)
    class0 = df[df['GT'] == 0].sort_values('avg_prob').reset_index(drop=True).add_suffix('_0')
    class1 = df[df['GT'] == 1].sort_values('avg_prob').reset_index(drop=True).add_suffix('_1')
    csv_path = os.path.join(save_dir, "ensemble_probabilities.csv")
    pd.concat([class0, class1], axis=1).to_csv(csv_path, index=False)
    print(f"Saved sorted probabilities to: {csv_path}")


if __name__ == "__main__":
    for fold_idx, val_ids in enumerate(VAL_FOLDS):
        fold = fold_idx + 1
        val_dataset = LymphedemaPairDataset_3frames(DATA_DIR, val_ids, TABULAR_FILE, transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        models = load_models(os.path.join(MODELS_DIR, f'Fold{fold}'))
        run_gradcam_ensemble(models, val_loader, save_dir=os.path.join(OUT_DIR, 'gradcam', f'Fold{fold}'))
