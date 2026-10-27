"""Unimodal-A+CL, stage 2 (paper Sec. III-C2).

The encoder of Unimodal-A is initialized with the weights pre-trained with the SupCon loss in stage 1
(train.py with the [Unimodal-A+CL pre-train] lines) and kept frozen. The MLP (dropout 0.5) and the classifier
are fine-tuned with the weighted BCE loss.
"""
import datetime
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import (DATA_DIR, DEVICE, LOG_DIR, SAVE_DIR, TABULAR_FILE, VAL_FOLDS, augment_pair, get_train_ids,
                    sample_names, set_seed, transform)
from dataset import LymphedemaPairDataset_3frames
from model import UnimodalA_CL
from train import NUM_EPOCHS, WEIGHT_DECAY, train_validate

EXPERIMENT = 'unimodal_a_cl'
PRETRAINED_DIR = os.path.join(SAVE_DIR, 'unimodal_a_cl_pretrain')  # output of stage 1
NUM_RUNS = 5
TRAIN_BATCH_SIZE, VAL_BATCH_SIZE = 12, 10

# Hyperparameters of Table III (Unimodal-A+CL)
POS_WEIGHT, LR, GAMMA, STEP_SIZE = 1.4, 1e-5, 0.9, 20


def run_batch(model, batch, criterion, train):
    image1, image2, labels, img1_names, img2_names, *_ = batch
    image1, image2 = image1.to(DEVICE), image2.to(DEVICE)
    labels = labels.to(DEVICE).view(-1).float()
    logits, _ = model(image1, image2)
    return logits, criterion(logits, labels), labels, sample_names(img1_names, img2_names)


def load_pretrained_encoder(fold):
    """Encoder weights of the stage-1 model of this fold."""
    paths = sorted(glob.glob(os.path.join(PRETRAINED_DIR, f'Fold{fold}', '*.pth')))
    if not paths:
        raise FileNotFoundError(f"No pre-trained model found in {PRETRAINED_DIR}/Fold{fold}. Run stage 1 first.")
    print(f"Loading encoder weights from {paths[0]}")
    state_dict = torch.load(paths[0], map_location='cpu')['model_state_dict']
    prefix = 'feature_extractor.'
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


if __name__ == "__main__":
    for fold_idx, val_ids in enumerate(VAL_FOLDS):
        fold = fold_idx + 1
        print(f"Training for Fold {fold}")
        train_ids = get_train_ids(val_ids)

        train_dataset = LymphedemaPairDataset_3frames(DATA_DIR, train_ids, TABULAR_FILE, transform, augment=augment_pair)
        val_dataset = LymphedemaPairDataset_3frames(DATA_DIR, val_ids, TABULAR_FILE, transform)
        print(f"Train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)

        encoder_weights = load_pretrained_encoder(fold)
        save_dir = os.path.join(SAVE_DIR, EXPERIMENT, f'Fold{fold}')
        os.makedirs(save_dir, exist_ok=True)

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx
            print(f"Run {run_idx + 1}, Seed: {seed}")
            set_seed(seed)

            model = UnimodalA_CL()
            model.feature_extractor.load_state_dict(encoder_weights)
            model = model.to(DEVICE)

            print("{:<60} | {:<10}".format("Parameter", "Trainable"))
            for name, param in model.named_parameters():
                print(f"{name:<60} | {param.requires_grad}")

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

            tt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, EXPERIMENT, f'fold{fold}_run{run_idx + 1}_{tt}'))
            checkpoint = train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, writer,
                                        num_epochs=NUM_EPOCHS, batch_fn=run_batch)
            writer.close()

            checkpoint.update({"model_summary": str(model), "learning_rate": LR, "scheduler_step_size": STEP_SIZE,
                               "scheduler_gamma": GAMMA, "pos_weight": POS_WEIGHT, "seed": seed})
            torch.save(checkpoint, os.path.join(save_dir, f'best_model_fold{fold}_run{run_idx + 1}_seed{seed}_{tt}.pth'))
