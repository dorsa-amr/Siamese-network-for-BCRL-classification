"""Datasets for the POCUS BCRL study (paper Sec. III-B).

Images are expected in `<parent_folder>/{no_lymphedema, lymphedema}/` and named `USxxx_<side><landmark>_<frame>.jpeg`,
e.g. `US019_L2_a.jpeg` (subject US019, left arm, landmark 2, first of the three frames).
Label 1: mild to severe lymphedema (severity > 1). Label 0: no or very mild lymphedema (severity <= 1).
"""
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LANDMARKS = ['2', '4']
TABULAR_COLUMNS = [
    'Perometer difference volume calibrated',  # calibrated arm volume difference
    'Total nodes removed',                     # number of dissected lymph nodes
    'BMI',
    'Time after surgery',                      # years
]


def load_tabular_data(info_file):
    """Reads the clinical data (one row per subject, indexed by subject ID)."""
    df = pd.read_excel(info_file, sheet_name='Sheet1')
    df['ID'] = df['ID'].astype(str)
    # Missing numbers of dissected nodes are replaced by the median of the subject's class
    medians = df.groupby('Label')['Total nodes removed'].median()
    df['Total nodes removed'] = df['Total nodes removed'].fillna(df['Label'].map(medians))
    return df.set_index('ID')


def tabular_features(df):
    """{subject ID: 4-D tabular feature vector}."""
    return {pid: torch.tensor(row[TABULAR_COLUMNS].to_numpy(dtype=float), dtype=torch.float32)
            for pid, row in df.iterrows()}


def _collect_images(parent_folder, subfolder, participant_ids):
    """{subject ID: {'L2': [paths], 'R2': [...], 'L4': [...], 'R4': [...]}} for one class folder."""
    images = {pid: {f'{side}{lm}': [] for side in 'LR' for lm in LANDMARKS} for pid in participant_ids}
    folder_path = os.path.join(parent_folder, subfolder)
    for img_name in sorted(os.listdir(folder_path)):
        if not img_name.endswith('.jpeg'):
            continue
        pid, region = img_name[:5], img_name[6:8]
        if pid in images and region in images[pid]:
            images[pid][region].append(os.path.join(folder_path, img_name))
    return images


class LymphedemaPairDataset_3frames(Dataset):
    """Siamese input (Sec. III-C): each of the three left-arm frames of a landmark is paired with each of the
    three right-arm frames of the same landmark (3x3 = 9 pairs per landmark, landmarks 2 and 4).
    Set `return_tabular=True` to also return the 4-D tabular feature vector (Multimodal model)."""

    def __init__(self, parent_folder, participant_ids, info_file, transform=None, augment=None, return_tabular=False):
        self.transform = transform
        self.augment = augment
        self.return_tabular = return_tabular
        self.tabular = tabular_features(load_tabular_data(info_file))
        self.image_pairs = []
        for label, subfolder in enumerate(['no_lymphedema', 'lymphedema']):
            for pid, regions in _collect_images(parent_folder, subfolder, participant_ids).items():
                for lm in LANDMARKS:
                    for l_img in regions[f'L{lm}']:
                        for r_img in regions[f'R{lm}']:
                            self.image_pairs.append((l_img, r_img, label))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.image_pairs[idx]
        img1, img2 = Image.open(img1_path), Image.open(img2_path)
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        if self.augment:
            img1, img2 = self.augment(img1, img2)

        pid = os.path.basename(img1_path)[:5]
        label = torch.tensor(label, dtype=torch.float32)
        names = (os.path.basename(img1_path), os.path.basename(img2_path), pid)
        if self.return_tabular:
            return (img1, img2, self.tabular[pid], label) + names
        return (img1, img2, label) + names


class LymphedemaSingleDataset_3frames(Dataset):
    """Single-arm input for the Unimodal-no-Siamese model (Sec. IV): the affected arm for class 1 subjects,
    and a randomly chosen (fixed) arm for class 0 subjects; three frames from landmarks 2 and 4."""

    def __init__(self, parent_folder, participant_ids, info_file, transform=None, augment=None):
        self.transform = transform
        self.augment = augment
        affected_side = load_tabular_data(info_file)['Affected side'].to_dict()
        self.image_list = []
        for label, subfolder in enumerate(['no_lymphedema', 'lymphedema']):
            for pid, regions in _collect_images(parent_folder, subfolder, participant_ids).items():
                if not any(regions.values()):
                    continue
                side = affected_side.get(pid) if label == 1 else random.choice(['L', 'R'])
                if side not in ['L', 'R']:
                    continue
                for lm in LANDMARKS:
                    for img_path in sorted(regions[f'{side}{lm}']):
                        self.image_list.append((img_path, label, pid))

        n_pos = sum(label for _, label, _ in self.image_list)
        print(f"Loaded {len(self.image_list)} images ({n_pos} lymphedema, {len(self.image_list) - n_pos} no-lymphedema)")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path, label, pid = self.image_list[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        if self.augment:
            img = self.augment(img)
        return img, torch.tensor(label, dtype=torch.float32), os.path.basename(img_path), pid


class LymphedemaDataset_Tabular_Only(Dataset):
    """Tabular data only (Unimodal-B): one sample per subject."""

    def __init__(self, info_file, participant_ids):
        df = load_tabular_data(info_file)
        self.tabular = tabular_features(df)
        self.labels = df['Label'].to_dict()
        self.participant_ids = [pid for pid in df.index if pid in participant_ids]

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        pid = self.participant_ids[idx]
        return self.tabular[pid], torch.tensor(self.labels[pid], dtype=torch.float32), pid
