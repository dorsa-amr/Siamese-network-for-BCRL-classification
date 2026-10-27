"""Shared settings and helpers: paths, cross-validation folds, image pre-processing,
data augmentation and evaluation metrics (paper Sec. III-B and III-C)."""
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Paths
DATA_DIR = './dataset/folders/3_frame/severity1_as_class0'  # B-mode frames: {no_lymphedema, lymphedema}/USxxx_{L|R}{2|4}_{a|b|c}.jpeg
TABULAR_FILE = './dataset/Id_info_uploaded.xlsx'           # clinical (tabular) data, one row per subject
SAVE_DIR = './checkpoints'
LOG_DIR = './runs'
OUT_DIR = './outputs'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Three-fold cross-validation: 19 subjects per validation fold, no subject in both training and validation
VAL_FOLDS = [
    ['US065', 'US053', 'US025', 'US051', 'US038', 'US019', 'US044', 'US027', 'US016', 'US054',
     'US033', 'US037', 'US022', 'US012', 'US050', 'US010', 'US061', 'US039', 'US059'],
    ['US066', 'US063', 'US046', 'US055', 'US032', 'US056', 'US047', 'US021', 'US015', 'US013',
     'US020', 'US034', 'US045', 'US043', 'US018', 'US011', 'US026', 'US035', 'US062'],
    ['US058', 'US060', 'US041', 'US052', 'US049', 'US040', 'US048', 'US017', 'US024', 'US014',
     'US030', 'US028', 'US029', 'US042', 'US023', 'US031', 'US036', 'US064', 'US009'],
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_participant_ids(data_dir=DATA_DIR):
    """Subject IDs (first 5 characters of the file names) found in the image folders."""
    ids = set()
    for subfolder in ['no_lymphedema', 'lymphedema']:
        for img_name in os.listdir(os.path.join(data_dir, subfolder)):
            if img_name.endswith('.jpeg'):
                ids.add(img_name[:5])
    return sorted(ids)


def get_train_ids(val_ids, data_dir=DATA_DIR):
    return [pid for pid in get_participant_ids(data_dir) if pid not in val_ids]


# Resize to 448x448, rescale to [0, 1], repeat the grayscale channel 3 times and standardize
# with ImageNet statistics (Sec. III-B1)
IMAGE_SIZE = (448, 448)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Data augmentation (Sec. III-B1): random cropping up to 5% and random rotation up to +-5 degrees,
# applied identically to the left- and right-arm images of a pair, plus Gaussian noise (std 0.001)
CROP_FRACTION = 0.95


def _random_crop_box(height, width):
    crop_h, crop_w = int(CROP_FRACTION * height), int(CROP_FRACTION * width)
    top = np.random.randint(0, height - crop_h + 1)
    left = np.random.randint(0, width - crop_w + 1)
    return top, left, crop_h, crop_w


def _augment(images):
    """Applies the same random crop and rotation to all images in the list and adds independent noise."""
    height, width = images[0].shape[-2:]
    if np.random.random() > 0.5:
        box = _random_crop_box(height, width)
        images = [TF.resized_crop(img, *box, [height, width], interpolation=TF.InterpolationMode.BILINEAR)
                  for img in images]
    if np.random.random() > 0.5:
        angle = np.random.randint(-5, 6)
        images = [TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR) for img in images]
    return [img + 0.001 * torch.randn_like(img) for img in images]


def augment_pair(img1, img2):
    img1, img2 = _augment([img1, img2])
    return img1, img2


def augment_single(img):
    return _augment([img])[0]


def sample_names(img1_names, img2_names=None):
    """Unique name of each sample, e.g. 'US019_L2_a_R2_b' for a pair or 'US019_L2_a' for a single image.
    Used to match predictions of the same sample across runs for ensembling."""
    if img2_names is None:
        return [str(name) for name in img1_names]
    return [f"{l.replace('.jpeg', '')}_{r.replace('.jpeg', '').split('_', 1)[1]}"
            for l, r in zip(img1_names, img2_names)]


def compute_metrics(labels, probs, preds=None):
    """Classification metrics; predictions default to probability > 0.5."""
    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()
    preds = (probs > 0.5).astype(int) if preds is None else np.asarray(preds).ravel()
    return {
        'accuracy': accuracy_score(labels, preds),
        'auc': roc_auc_score(labels, probs),
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
    }


def log_metrics(writer, split, loss, metrics, epoch):
    writer.add_scalar(f'Loss/{split}', loss, epoch)
    writer.add_scalar(f'Accuracy/{split}', metrics['accuracy'], epoch)
    writer.add_scalar(f'AUC_Score/{split}', metrics['auc'], epoch)
    writer.add_scalar(f'F1_Score/{split}', metrics['f1'], epoch)
    writer.add_scalar(f'Precision/{split}', metrics['precision'], epoch)
    writer.add_scalar(f'Recall/{split}', metrics['recall'], epoch)
