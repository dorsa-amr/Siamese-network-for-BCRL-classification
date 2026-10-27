"""Unimodal-B: three-fold cross-validation training on the tabular clinical data only (paper Sec. III-C)."""
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import DEVICE, LOG_DIR, SAVE_DIR, TABULAR_FILE, VAL_FOLDS, get_train_ids, set_seed
from dataset import LymphedemaDataset_Tabular_Only
from model import UnimodalB
from train import NUM_EPOCHS, WEIGHT_DECAY, train_validate

EXPERIMENT = 'unimodal_b'
NUM_RUNS = 5
TRAIN_BATCH_SIZE, VAL_BATCH_SIZE = 12, 10

# Hyperparameters of Table III (Unimodal-B)
POS_WEIGHT, LR, GAMMA, STEP_SIZE = 1.1, 1e-4, 0.6, 50


def run_batch(model, batch, criterion, train):
    x_tab, labels, pids = batch
    x_tab, labels = x_tab.to(DEVICE), labels.to(DEVICE).float()
    if train:
        x_tab = x_tab + 0.01 * torch.randn_like(x_tab)  # Gaussian noise on the tabular features (Sec. III-B2)
    logits = model(x_tab)
    return logits, criterion(logits, labels), labels, [str(pid) for pid in pids]


if __name__ == "__main__":
    for fold_idx, val_ids in enumerate(VAL_FOLDS):
        fold = fold_idx + 1
        print(f"Training for Fold {fold}")
        train_ids = get_train_ids(val_ids)

        train_dataset = LymphedemaDataset_Tabular_Only(TABULAR_FILE, train_ids)
        val_dataset = LymphedemaDataset_Tabular_Only(TABULAR_FILE, val_ids)
        print(f"Train subjects: {len(train_dataset)}, validation subjects: {len(val_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)

        save_dir = os.path.join(SAVE_DIR, EXPERIMENT, f'Fold{fold}')
        os.makedirs(save_dir, exist_ok=True)

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx
            print(f"Run {run_idx + 1}, Seed: {seed}")
            set_seed(seed)

            model = UnimodalB().to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

            tt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, EXPERIMENT, f'fold{fold}_run{run_idx + 1}_{tt}'))
            checkpoint = train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, writer,
                                        num_epochs=NUM_EPOCHS, batch_fn=run_batch)
            writer.close()

            checkpoint.update({"model_summary": str(model), "learning_rate": LR, "scheduler_step_size": STEP_SIZE,
                               "scheduler_gamma": GAMMA, "pos_weight": POS_WEIGHT, "seed": seed})
            torch.save(checkpoint, os.path.join(save_dir, f'best_model_fold{fold}_run{run_idx + 1}_seed{seed}_{tt}.pth'))
