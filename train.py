"""Three-fold cross-validation training of the image-based models (paper Sec. III-C).

The same script trains several methods of the paper; the active lines train Unimodal-A. Switchable lines end
with a tag and switchable blocks start with a tag line. To train a method, uncomment every line/block whose tag
contains that method and comment out the other tagged lines/blocks:
    [Unimodal-A]               Siamese network, images only
    [Multimodal]               Siamese network, images + tabular data
    [Unimodal-no-Siamese]      single-arm network
    [Unimodal-A+CL pre-train]  stage 1 of Unimodal-A+CL: SupCon pre-training of the encoder
                               (stage 2 is train_contrastive_finetune.py)
Unimodal-B (tabular data only) is trained with train_tabular_only.py.
"""
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import (DATA_DIR, DEVICE, LOG_DIR, SAVE_DIR, TABULAR_FILE, VAL_FOLDS, augment_pair, augment_single,
                    compute_metrics, get_train_ids, log_metrics, sample_names, set_seed, transform)
from dataset import LymphedemaPairDataset_3frames, LymphedemaSingleDataset_3frames
from model import Multimodal, UnimodalA, UnimodalA_CL, UnimodalNoSiamese

EXPERIMENT = 'unimodal_a'                      # [Unimodal-A]
# EXPERIMENT = 'multimodal'                    # [Multimodal]
# EXPERIMENT = 'unimodal_no_siamese'           # [Unimodal-no-Siamese]
# EXPERIMENT = 'unimodal_a_cl_pretrain'        # [Unimodal-A+CL pre-train]

NUM_RUNS = 5                                   # [Unimodal-A] [Multimodal] [Unimodal-no-Siamese] five seeds per fold (Sec. IV)
# NUM_RUNS = 1                                 # [Unimodal-A+CL pre-train] one pre-trained encoder per fold
NUM_EPOCHS = 200
TRAIN_BATCH_SIZE, VAL_BATCH_SIZE = 12, 10      # [Unimodal-A] [Multimodal] [Unimodal-no-Siamese]
# TRAIN_BATCH_SIZE, VAL_BATCH_SIZE = 60, 30    # [Unimodal-A+CL pre-train] larger batches give more positives/negatives per anchor

# Hyperparameters of Table III: positive class weight p_c, base learning rate, gamma and step size s of the scheduler
POS_WEIGHT, LR, GAMMA, STEP_SIZE = 1.4, 1e-5, 0.9, 20     # [Unimodal-A] [Unimodal-no-Siamese] [Unimodal-A+CL pre-train]
# POS_WEIGHT, LR, GAMMA, STEP_SIZE = 1.1, 1e-5, 0.9, 20   # [Multimodal]
WEIGHT_DECAY = 1e-5

supcon_loss = losses.SupConLoss(temperature=0.1)  # SupCon loss, Eq. (2), tau = 0.1


def run_batch(model, batch, criterion, train):
    """Forward pass and loss for one batch. Returns logits, loss, labels and sample names."""
    # [Unimodal-A] [Unimodal-A+CL pre-train]
    image1, image2, labels, img1_names, img2_names, *_ = batch
    image1, image2 = image1.to(DEVICE), image2.to(DEVICE)
    # # [Multimodal]
    # image1, image2, x_tab, labels, img1_names, img2_names, *_ = batch
    # image1, image2, x_tab = image1.to(DEVICE), image2.to(DEVICE), x_tab.to(DEVICE)
    # if train:
    #     x_tab = x_tab + 0.01 * torch.randn_like(x_tab)  # Gaussian noise on the tabular features (Sec. III-B2)
    # # [Unimodal-no-Siamese]
    # image1, labels, img1_names, *_ = batch
    # image1, img2_names = image1.to(DEVICE), None
    labels = labels.to(DEVICE).view(-1).float()

    logits = model(image1, image2)                          # [Unimodal-A]
    # logits = model(image1, image2, x_tab)                 # [Multimodal]
    # logits = model(image1)                                # [Unimodal-no-Siamese]
    # logits, contrast_features = model(image1, image2)     # [Unimodal-A+CL pre-train]

    loss = criterion(logits, labels)                        # weighted BCE loss, Eq. (1)
    # loss = supcon_loss(contrast_features, labels)         # [Unimodal-A+CL pre-train]

    return logits, loss, labels, sample_names(img1_names, img2_names)


def validate(model, val_loader, criterion, batch_fn=run_batch):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_names = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, loss, labels, names = batch_fn(model, batch, criterion, train=False)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_names.extend(names)
    return total_loss / len(val_loader), np.array(all_labels), np.array(all_probs), all_names


def train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, writer,
                   num_epochs=NUM_EPOCHS, batch_fn=run_batch):
    """Trains for `num_epochs` and returns the checkpoint of the epoch with the lowest validation loss.
    `batch_fn(model, batch, criterion, train)` returns logits, loss, labels and sample names of a batch."""
    best_val_loss = float('inf')
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch"):
        model.train()
        total_loss = 0.0
        all_labels, all_probs = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, loss, labels, _ = batch_fn(model, batch, criterion, train=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_metrics = compute_metrics(all_labels, all_probs)
        log_metrics(writer, 'Train', train_loss, train_metrics, epoch)

        val_loss, val_labels, val_probs, val_names = validate(model, val_loader, criterion, batch_fn)
        val_metrics = compute_metrics(val_labels, val_probs)
        log_metrics(writer, 'Validation', val_loss, val_metrics, epoch)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "best_val_epoch": epoch,
                "best_val_loss": val_loss,
                "best_val_auc": val_metrics['auc'],
                "best_val_acc": val_metrics['accuracy'],
                "bestepoch_f1_score_val": val_metrics['f1'],
                "bestepoch_prec_val": val_metrics['precision'],
                "bestepoch_rec_val": val_metrics['recall'],
                "bestepoch_probs": val_probs.reshape(1, -1),
                "bestepoch_preds": (val_probs > 0.5).astype(float).reshape(1, -1),
                "bestepochs_labels": val_labels.reshape(1, -1),
                "bestepoch_img_names": val_names,
            }

        tqdm.write(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                   f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        scheduler.step()

    print(f"Best epoch {checkpoint['best_val_epoch']}: val loss = {checkpoint['best_val_loss']:.4f}, "
          f"AUC = {checkpoint['best_val_auc']:.4f}, accuracy = {checkpoint['best_val_acc']:.4f}, "
          f"F1 = {checkpoint['bestepoch_f1_score_val']:.4f}, precision = {checkpoint['bestepoch_prec_val']:.4f}, "
          f"recall = {checkpoint['bestepoch_rec_val']:.4f}")
    return checkpoint


if __name__ == "__main__":
    for fold_idx, val_ids in enumerate(VAL_FOLDS):
        fold = fold_idx + 1
        print(f"Training for Fold {fold}")
        train_ids = get_train_ids(val_ids)

        set_seed(42)
        # [Unimodal-A] [Unimodal-A+CL pre-train]
        train_dataset = LymphedemaPairDataset_3frames(DATA_DIR, train_ids, TABULAR_FILE, transform, augment=augment_pair)
        val_dataset = LymphedemaPairDataset_3frames(DATA_DIR, val_ids, TABULAR_FILE, transform)
        # # [Multimodal]
        # train_dataset = LymphedemaPairDataset_3frames(DATA_DIR, train_ids, TABULAR_FILE, transform, augment=augment_pair, return_tabular=True)
        # val_dataset = LymphedemaPairDataset_3frames(DATA_DIR, val_ids, TABULAR_FILE, transform, return_tabular=True)
        # # [Unimodal-no-Siamese]
        # train_dataset = LymphedemaSingleDataset_3frames(DATA_DIR, train_ids, TABULAR_FILE, transform, augment=augment_single)
        # val_dataset = LymphedemaSingleDataset_3frames(DATA_DIR, val_ids, TABULAR_FILE, transform)
        print(f"Train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)

        save_dir = os.path.join(SAVE_DIR, EXPERIMENT, f'Fold{fold}')
        os.makedirs(save_dir, exist_ok=True)

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx
            print(f"Run {run_idx + 1}, Seed: {seed}")
            set_seed(seed)

            model = UnimodalA()                        # [Unimodal-A]
            # model = Multimodal()                     # [Multimodal]
            # model = UnimodalNoSiamese()              # [Unimodal-no-Siamese]
            # model = UnimodalA_CL(pretrain=True)      # [Unimodal-A+CL pre-train]
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
            checkpoint = train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, writer)
            writer.close()

            checkpoint.update({"model_summary": str(model), "learning_rate": LR, "scheduler_step_size": STEP_SIZE,
                               "scheduler_gamma": GAMMA, "pos_weight": POS_WEIGHT, "seed": seed})
            torch.save(checkpoint, os.path.join(save_dir, f'best_model_fold{fold}_run{run_idx + 1}_seed{seed}_{tt}.pth'))
