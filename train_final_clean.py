#%% Imports, Classes
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR,CyclicLR,ExponentialLR
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from model import UNext_Wrapper
from model import ResNet
from model import SwinV2Model
from model import ResNet_Supervised
from model import MobileNetBinaryClassifier
from model import MobileNet_Supervised
from model import ResNet50_Dino_Supervised
from model import ResNet50_Dino_Supervised_ImageTabular, ResNet50_Dino_Supervised_Image_Only, Model_Tabular_Only, ResNet50_Dino_Supervised_Single
from model import ResNet50_Dino_Supervised_Image_Only_contrastive
from model import ResNet50_Dino
from model import DinoV2_Supervised
from model import twopairs_ResNet50_Dino_Supervised
from model import MLP_Encoder_Image_Only, CNN_Encoder_Image_Only
from loss import ContrastiveLoss
from loss import ContrastiveLoss_weighted
from loss import FocalLoss2
from pytorch_metric_learning import losses
from dataset import LymphedemaPairDataset, LymphedemaPairDataset_3frames, LymphedemaSingleDataset_3frames
from dataset import LymphedemaPairDataset2
from dataset import SingleLymphedemaDataset
from dataset import TwopairsLymphedemaPairDataset
from collections import defaultdict
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import vit_small_patch16_224
import random
from torchsummary import summary

def set_seed(seed):
    """
    Function to set the random seed for reproducibility across various libraries.
    """
    # Set seed for Python random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if you're using CUDA)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior (slow, but reproducible)
    torch.backends.cudnn.benchmark = False  # Disable non-deterministic behavior in cudnn

#SET SEED HERE    
set_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Data augmentations and visualization
def my_rotate(image):
    if np.random.random() > 0.5:
        angle = np.random.randint(-5, 6)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR) 
    return image

def flip_horizontally(image):
    if np.random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
    return image

def my_random_noise(image):
    noise = 0.001 * torch.randn_like(image)
    image = image + noise
    return image

def my_crop(image, output_size=(448,448)):
    # # Crop the dots on the right side of the image
    # image = TF.crop(image, 0, 0, image.shape[-2] - 5, image.shape[-1] - 8)

    if np.random.random() > 0.5:
        orig_h, orig_w = image.shape[-2], image.shape[-1]
        crop_h, crop_w = (213, 213)

        # Randomly select top-left corner for cropping
        top = np.random.randint(0, orig_h - crop_h + 1)
        left = np.random.randint(0, orig_w - crop_w + 1)

        # Crop the 2D image (H, W)
        image = TF.crop(image, top, left, crop_h, crop_w)
        #print ('cropped image size:', image.shape)
        
    # Resize the cropped image back to output size
    image = TF.resize(image, output_size, interpolation=TF.InterpolationMode.BILINEAR)
    return image

def my_augmentations(img1, img2):
    img1, img2 = my_crop(img1),my_crop(img2)
    img1, img2 = my_rotate(img1), my_rotate(img2)
    # img2 = flip_horizontally(img2)
    img1, img2 = my_random_noise(img1),my_random_noise(img2)
    return img1, img2


# def my_augmentations(img):
#     img = my_crop(img)
#     img = my_rotate(img)
#     # img = flip_horizontally(img)
#     img = my_random_noise(img)
#     return img

# def val_crop(image, output_size=(224,224)):
#     # Crop the dots on the right side of the image
#     image = TF.crop(image, 0, 0, image.shape[-2] - 5, image.shape[-1] - 8)
#     image = TF.resize(image, output_size, interpolation=TF.InterpolationMode.BILINEAR)
#     return image

# def val_flip_horizontally(image):
#     image = torch.flip(image, dims=[-1])
#     return image

def val_augmentations(img1, img2):
    # img1, img2 = val_crop(img1),val_crop(img2)
    # img2 = val_flip_horizontally(img2)
    return img1, img2


# def val_augmentations(img):
#     # img1, img2 = val_crop(img1),val_crop(img2)
#     # img2 = val_flip_horizontally(img2)
#     return img


# def my_augmentations(img1, img2, img3, img4):
#     img1, img2, img3, img4 = my_crop(img1), my_crop(img2), my_crop(img3), my_crop(img4)
#     img1, img2, img3, img4 = my_rotate(img1), my_rotate(img2), my_rotate(img3), my_rotate(img4)
#     img1, img2, img3, img4 = flip_horizontally(img1),flip_horizontally(img2), flip_horizontally(img3),flip_horizontally(img4)
#     img1, img2, img3, img4 = my_random_noise(img1),my_random_noise(img2), my_random_noise(img3),my_random_noise(img4)
#     return img1, img2, img3, img4

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

transform = transforms.Compose([
    transforms.Resize((448,448)),  # Doubling the size (448, 448) 224,224
    transforms.ToTensor(),  # Converts to [0, 1] range
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Duplicate the single channel to 3 channels for not needing to train the first conv layer
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])


def data_visualization(dataset):
    for idx in range(len(dataset)): #range(round(len(dataset)/2))
        output = dataset[idx]

        # img1, img2, x_tab, label, img1_path, img2_path, participant_id = output
        img1, img2, label, img1_path, img2_path, participant_id = output
        #print(img1.shape, img2.shape, label, participant_id)

        img1_np = img1.permute(1, 2, 0).numpy() if isinstance(img1, torch.Tensor) else img1
        img2_np = img2.permute(1, 2, 0).numpy() if isinstance(img2, torch.Tensor) else img2

        img1_first_channel = img1_np[..., 0]
        img2_first_channel = img2_np[..., 0]

        img1_first_channel = img1_first_channel*0.229 + 0.485 #Denormalizes an image tensor (C, H, W) or numpy array
        img2_first_channel = img2_first_channel*0.229 + 0.485 #Denormalizes an image tensor (C, H, W) or numpy array

        #  # Clip values to [0, 1] range
        # img1_first_channel = np.clip(img1_first_channel, 0, 1)
        # img2_first_channel = np.clip(img2_first_channel, 0, 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img1_first_channel, cmap='gray')
        axes[0].set_title(f"{img1_path}")
        axes[1].imshow(img2_first_channel, cmap='gray')
        axes[1].set_title(f"{img2_path}")

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.show() #block=False


#%% Train and validation functions


def validate_supervised(model, val_loader, criterion, device, writer=None, epoch=None): #criterion=None,
    model.eval() 
    total_loss = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    correct = 0
    total_samples = 0

    all_labels = []
    all_probs = []
    all_preds = []
    all_logits = []
    all_img_names = []
    with torch.no_grad(): 
        for co,batch in enumerate(val_loader):
            # image1, image2, x_tab , labels, img1_names, img2_names, *_ = batch
            # image1, image2, x_tab, labels = [x.to(device) for x in (image1, image2, x_tab, labels)]
            # labels.float() #.unsqueeze(1)  # Ensure labels shape (batch_size, 1)  # Ensure float labels 

            image1, image2, labels, img1_names, img2_names, *_ = batch
            image1, image2, labels = image1.to(device), image2.to(device),labels.to(device) #.unsqueeze(1) 
            labels = labels.view(-1).float()

            # image1, labels, img1_names, *_ = batch
            # image1, labels = image1.to(device),labels.to(device) #.unsqueeze(1) 
            # labels = labels.view(-1).float()
            
            # #Normalizing tabular data (Values were caculated offline in Id_info.xlsx. For NaNs, median values were set (5 for label 0s and 13 for label 1s))
            # s1 = x_tab.shape   
            # m = torch.zeros(s1).cuda()
            # st = torch.zeros(s1).cuda()
            # m[:,0] = 8.7000
            # m[:,1] = 9.8148
            # m[:,2] = 27.0305
            # m[:,3] = 5.3449
            # st[:,0] = 13.6800
            # st[:,1] = 6.0659
            # st[:,2] = 4.4998
            # st[:,3] = 6.6284
            # x_tab = (x_tab-m)/st

            # logits = model(image1,image2,x_tab)  # Forward pass (logits shape: [batch_size, 1]) #, contrast_features
            logits = model(image1,image2) #,  contrast_features
            # logits,  contrast_features = model(image1,image2) #
            # logits = model(x_tab)
            # logits = model(image1)

            # print('lable sizes: ', labels.shape)
            # print('logits sizes: ', logits.shape)

            all_logits.extend(logits.cpu().numpy())

            # loss_contrastive = soft_triplet_loss(feauters,labels)
            # Sup_con_loss = losses.SupConLoss(temperature=0.1)
            # loss_contrastive = Sup_con_loss(contrast_features,labels)
            loss_bce = criterion(logits,labels)

            # loss_v = loss_bce + 0*loss_contrastive
            loss_v = loss_bce 
            # loss_v = loss_contrastive

            # criterion1 = nn.BCEWithLogitsLoss()
            # loss_v1 = criterion1(logits,labels)
            # loss_v2 = Soft_Triplet_Loss(embeds, labels.squeeze(1)>0)
            # loss_v = 1*loss_v1 + 3*loss_v2

            total_loss += loss_v.item() 
            # total_loss2 += loss_contrastive.item() 
            # total_loss3 += loss_bce.item() 

            #* images.size(0)  # Accumulate total loss

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions
            preds = (probs > 0.5).float()

            # Store results for AUC & F1 computation
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # img1_names = [os.path.splitext(os.path.basename(path))[0] for path in img1_path]
            # img_names = [name + ('_R2' if 'L2' in name else '_R4' if 'L4' in name else '') for name in img1_names]
            

            # print('IMAGE 1 NAMES:', img1_names)
            # print('IMAGE 2 NAMES:', img2_names)
            # Ensure they're plain strings (if needed)
            img1_names = [str(name) for name in img1_names]
            # all_img_names.extend(img1_names)
            # combined_names = [
            #     f"{l.replace('.jpeg', '')}_{r.replace('.jpeg', '').split('_', 1)[1]}"
            #     for l, r in zip(img1_names, img2_names)
            # ]
            # all_img_names.extend(combined_names)
            all_img_names.extend(img1_names)


            # print('COMBINED NAMES:', combined_names)

            # Compute batch-wise accuracy
            # correct += (preds == labels).sum().item()
            # total_samples += labels.size(0)

    all_probs_np = np.array(all_probs).reshape(1,-1)
    all_preds_np = np.array(all_preds).reshape(1,-1)
    all_labels_np = np.array(all_labels).reshape(1,-1)

    print('val All probs: ', all_probs_np)
    print('Val All preds:  ', all_preds_np)
    print('Val All labels: ', all_labels_np)
    print('Val All image names: ', all_img_names)
    print('Val All labels size: ',len(all_labels_np[0]))
    
    avg_loss = total_loss / (co+1)  # Compute average loss
    # contrast_loss = total_loss2 / (co+1) 
    # BCE_loss = total_loss3 / (co+1) 
    
    # print('total loss: ', total_loss)
    # print('avg loss: ', avg_loss)
    # print('total samples: ', total_samples)

    # accuracy = correct / total_samples  # Compute accuracy

    # Compute AUC and F1 Score
    # Compute metrics for the training epoch
    accuracy = accuracy_score (all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)  # AUC needs raw probabilities
    f1 = f1_score(all_labels, all_preds)  # F1 uses binary predictions
    precision = precision_score(all_labels, all_preds)  #pos_label='positive_label')
    recall = recall_score(all_labels, all_preds) #, pos_label='positive_label')


    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
        # writer.add_scalar('contrast_Loss/Validation', contrast_loss, epoch)
        # writer.add_scalar('BCE_Loss/Validation', BCE_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch) 
        writer.add_scalar('AUC_Score/Validation', auc_score, epoch)   
        writer.add_scalar('F1_Score/Validation', f1, epoch) 
        writer.add_scalar('Recall/Validation', recall, epoch) 
        writer.add_scalar('Precision/Validation', precision, epoch) 

    return avg_loss, accuracy, auc_score, f1, precision, recall, all_probs_np, all_preds_np, all_labels_np, all_logits, all_img_names

def train_validate_supervised(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, writer): # scheduler, criterion
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_auc = 0
    # # Initialize early stopping
    # early_stopping = EarlyStopping(patience=20, min_delta=0.0, verbose=True)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch", leave=True):
        
        # Train
        model.train()
        total_loss = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        correct = 0
        total_samples = 0 
        all_preds = []
        all_probs = []
        all_labels = []
     
        for co, batch in enumerate(train_loader): 
            # image1, image2, x_tab, labels, img1_names, img2_names, *_ = batch
            # image1, image2, x_tab, labels = image1, image2, x_tab, labels = [x.to(device) for x in (image1, image2, x_tab, labels)]
            # labels.float() #.unsqueeze(1)  # Ensure labels shape (batch_size, 1)  # Ensure float labels
            
            image1, image2, labels, img1_names, img2_names, *_ = batch
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device) #.unsqueeze(1) 
            labels = labels.view(-1).float()

            # image1, labels, img1_names, *_ = batch
            # image1, labels = image1.to(device),labels.to(device) #.unsqueeze(1) 
            # labels = labels.view(-1).float()
            
            # #Normalizing tabular data (Values were caculated offline in Id_info.xlsx. For NaNs, median values were set (5 for label 0s and 13 for label 1s))
            # s1 = x_tab.shape   
            # m = torch.zeros(s1).cuda()
            # st = torch.zeros(s1).cuda()
            # m[:,0] = 8.7000
            # m[:,1] = 9.8148
            # m[:,2] = 27.0305
            # m[:,3] = 5.3449
            # st[:,0] = 13.6800
            # st[:,1] = 6.0659
            # st[:,2] = 4.4998
            # st[:,3] = 6.6284
            # x_tab = (x_tab-m)/st

            # x_tab  = x_tab + 0.01*torch.randn_like(x_tab)

            optimizer.zero_grad()

            # logits = model(image1,image2,x_tab)  # Forward pass (logits shape: [batch_size, 1]) #, contrast_features 
            logits = model(image1,image2) #, contrast_features
            # logits, contrast_features = model(image1,image2) #
            # logits = model(x_tab)
            # logits, embeds = model(image1,image2, image3, image4)
            # logits = model(image1)
                           
            # print('lables size: ',labels.shape)
            # print('contrastive_features size: ',contrast_features.shape)
            # print('logits size: ',logits.shape)

            # loss_contrastive = soft_triplet_loss(feauters,labels)
            # Sup_con_loss = losses.SupConLoss(temperature=0.1)
            # loss_contrastive = Sup_con_loss(contrast_features,labels)
            loss_bce = criterion(logits,labels)

            # loss = loss_bce + 0*loss_contrastive
            loss = loss_bce 
            # loss = loss_contrastive

            # print(loss)
            # print(loss_contrastive)
            # criterion1 = nn.BCEWithLogitsLoss()
            # loss_1 = criterion1(logits,labels)
            # loss_2 = Soft_Triplet_Loss(embeds, labels.squeeze(1)>0)
            # loss = 1*loss_1 + 3*loss_2
            
            # print("Embeds size: ", embeds.shape)
            # print("Lables size: ", labels.shape)
            # loss = Soft_Triplet_Loss(embeds, labels.squeeze(1)>0) #Triplet

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()  # Accumulate loss
            # total_loss2 += loss_contrastive.item()
            # total_loss3 += loss_bce.item()

            # Convert logits to predictions
            probs = torch.sigmoid(logits).detach().cpu().numpy()  # Convert logits to probabilities
            preds = (probs > 0.5).astype(int) # Convert probabilities to binary predictions
            labels_np = labels.cpu().numpy()

            # correct += np.sum(preds == labels_np)
            # total_samples += labels_np.shape[0]

            # Store results for AUC & F1 computation
            all_labels.extend(labels_np)
            all_probs.extend(probs)
            all_preds.extend(preds)

        print('All probs: ', np.array(all_probs).reshape(1,-1))
        print('All preds: ', np.array(all_preds).reshape(1,-1))
        print('All labels: ', np.array(all_labels).reshape(1,-1))
        print('All labels size: ', len(all_labels))

        train_loss = total_loss / (co+1)   # Average loss
        # train_acc = correct / total_samples  # Accuracy
        # contrast_loss = total_loss2 / (co+1) 
        # BCE_loss = total_loss3 / (co+1) 

        # Compute metrics for the training epoch
        train_acc = accuracy_score (all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_probs)
        train_f1 = f1_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        # writer.add_scalar('contrast_Loss/Train', contrast_loss, epoch) 
        # writer.add_scalar('BCE_Loss/Train', BCE_loss, epoch)
        writer.add_scalar('AUC_Score/Train', train_auc, epoch)
        writer.add_scalar('F1_Score/Train', train_f1, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('Precision/Train', train_precision, epoch)

        #  Validation
        val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_probs, val_preds, val_labels, val_logits, img_names = validate_supervised(model, val_loader,criterion, device, writer, epoch) 
        # print('val labels: ', val_labels)
        # print('val outputs: ', val_logits)

        # if epoch == num_epochs - 1: 
        # if val_loss < best_val_loss:
            # best_val_loss = val_loss
            # best_val_loss_epoch = epoch
            # bestepoch_val_auc_score = val_auc
            # bestepoch_best_f1_score_val = val_f1
            # # bestepoch_best_acc_threshold_val = best_threshold_acc
            # bestepoch_accuracy_val = val_acc
            # model_best = model.state_dict()

        # if val_auc > best_val_auc:
        # if val_acc >= best_val_acc:
        # if epoch >= 200:
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_val_auc = val_auc
            # bestepoch_val_loss = val_loss
            bestepoch_f1_score_val = val_f1
            # bestepoch_best_acc_threshold_val = best_threshold_acc
            best_val_acc = val_acc
            bestepoch_prec_val = val_precision
            bestepoch_rec_val = val_recall
            bestepoch_probs = val_probs
            bestepoch_preds = val_preds
            bestepochs_labels= val_labels
            bestepoch_img_names = img_names

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_epoch": best_val_epoch,
                "best_val_loss": best_val_loss,
                "best_val_auc": best_val_auc,
                "best_val_acc": best_val_acc,
                "bestepoch_f1_score_val": bestepoch_f1_score_val,
                "bestepoch_prec_val": bestepoch_prec_val,
                "bestepoch_rec_val": bestepoch_rec_val,
                "bestepoch_probs": bestepoch_probs,
                "bestepoch_preds": bestepoch_preds,
                "bestepochs_labels": bestepochs_labels,
                "bestepoch_img_names": bestepoch_img_names
            }

        if epoch == num_epochs - 1:
            checkpoint_last = {
                "last_epoch_model_state_dict": model.state_dict(),
                "last_epoch_epoch": epoch,
                "last_epoch_val_loss": val_loss,
                "last_epoch_val_auc": val_auc,
                "last_epoch_val_acc": val_acc,
                "last_epoch_val_f1": val_f1,
                "last_epoch_val_precision": val_precision,
                "last_epoch_val_recall": val_recall,
                "last_epoch_val_probs": val_probs,
                "last_epoch_val_preds": val_preds,
                "last_epoch_val_labels": val_labels,
                "last_epoch_img_names": img_names
            }
            
        tqdm.write((
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
            f"Train F1 Score: {train_f1:.4f}, Val F1 Score: {val_f1:.4f}, "
            # f"Train F1 Score threshold: {best_threshold_acc:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}"
        ))

        # # scheduler.step(val_loss)
        scheduler.step()

        # log the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        # current_lr = optimizer.param_groups[0]['lr']
        
        # print(f"Current Learning Rate: {current_lr:.10f}")


    print(
    f"""
    best val loss epoch number = {best_val_epoch}
    best val loss = {best_val_loss:.4f}
    val auc score = {best_val_auc:.4f}
    val accuracy = {best_val_acc:.4f}
    val f1 score  = {bestepoch_f1_score_val:.4f}  
    precision  = {bestepoch_prec_val:.4f}
    recall  = {bestepoch_rec_val:.4f}
    val probabilities =  {bestepoch_probs}
    val predictions =  {bestepoch_preds}
    val labels       = {bestepochs_labels}
    images = {bestepoch_img_names}
    """
    )

    #, best threshold = {bestepoch_best_acc_threshold_val:.4f} " 
    
    return checkpoint, checkpoint_last 

#%% Main
# root_dir = './dataset/folders/1_frame/flipped_checked_data/severity1_as_class0'
root_dir = './dataset/folders/3_frame/severity1_as_class0'

participant_ids = set()
for subfolder in ['no_lymphedema', 'lymphedema']:
    folder_path = os.path.join(root_dir, subfolder)
    for img_name in os.listdir(folder_path):
        if img_name.endswith(".jpeg"):
            participant_id = img_name[:5]
            participant_ids.add(participant_id)

participant_ids = sorted(list(participant_ids)) 

# 33.33% val (Label 0 to severiy 1) (19 IDs in each fold from 57 IDs)
val_ids_f1 = ['US065', 'US053', 'US025','US051', 'US038', 'US019', 'US044', 'US027', 'US016', 
              'US054', 'US033', 'US037', 'US022', 'US012', 'US050', 'US010', 'US061','US039', 'US059'] #Added new: US039 from fold 3, US065,US061,US059
# val_ids_f1_severities = [2, 3.2, 4, 2, 2.5, 2, 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

val_ids_f2 = ['US066', 'US063', 'US046', 'US055', 'US032', 'US056', 'US047', 'US021', 'US015', 'US013', 
              'US020', 'US034', 'US045', 'US043', 'US018', 'US011', 'US026', 'US035', 'US062'] #Added new: US063,US062,US066
# val_ids_f2_severities = [2, 2, 3, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

val_ids_f3 = ['US058', 'US060', 'US041', 'US052', 'US049', 'US040', 'US048','US017', 'US024', 
              'US014', 'US030', 'US028', 'US029', 'US042', 'US023', 'US031', 'US036', 'US064', 'US009'] #Added new: ,US058 from fold1,US060,US064,US009
# val_ids_f3_severities = [2, 2, 4.5, 2.5, 2.38, 2, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

test_ids = []

# Choose the validation fold for each iteration
val_folds = [val_ids_f1, val_ids_f2, val_ids_f3] 
# val_folds = [val_ids_f3]

# Train and validate the model for each fold and each run
for fold_idx, val_fold_ids in enumerate(val_folds):
    print(f"Training for Fold {fold_idx + 1}")

    # Define train and validation IDs for this fold
    excluded_base_ids = set(id.replace('_REV', '') for id in val_fold_ids + test_ids)
    train_ids = [pid for pid in participant_ids if pid.replace('_REV', '') not in excluded_base_ids]
    # train_ids = [pid for pid in participant_ids if pid not in val_fold_ids + test_ids]
    print(train_ids)
    
    tabular_file = "./dataset/Id_info_uploaded.xlsx"

    train_dataset = LymphedemaPairDataset_3frames(root_dir, train_ids, tabular_file, transform=transform, augment=my_augmentations)
    val_dataset = LymphedemaPairDataset_3frames(root_dir, val_fold_ids, tabular_file, transform=transform, augment=val_augmentations)
    
    # train_dataset = LymphedemaSingleDataset_3frames(root_dir, train_ids, tabular_file, transform=transform, augment=my_augmentations)
    # val_dataset = LymphedemaSingleDataset_3frames(root_dir, val_fold_ids, tabular_file, transform=transform, augment=val_augmentations)
    

    print(f"Train dataset pairs: {len(train_dataset)}")
    print(f"Validation dataset pairs: {len(val_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=False) #12
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True) #10
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_img_names = []
    for batch in val_loader:
        img1s, img2s, labels, img1_names, img2_names, participant_ids = batch #img2s,img2_names, x_tabs,
    #     # print(f"Batch size: {len(labels)}")
    #     # print(f"Image 1 paths: {img1_paths}")
    #     # print(f"Image 2 paths: {img2_paths}")
        all_img_names.append(img1_names)
    #     # print(f"Tabular: {x_tabs}")
        print(f"Labels: {labels}")
        print(img1s.shape, labels.shape) #img2s.shape, x_tabs.shape, 
    print(all_img_names)



    # Now, for each run, set the seed and train the model
    for run_idx in range(5):  # 5 runs per fold
        seed = 42 + fold_idx * 5 + run_idx  # Unique seed for each fold and run
        # seed = 52 + fold_idx * 5 + run_idx  # Unique seed for each fold and run
        print(f"Run {run_idx + 1}, Seed: {seed}")

        # Set the seed for reproducibility
        set_seed(seed)

        # model = ResNet50_Dino_Supervised_ImageTabular()
        model = ResNet50_Dino_Supervised_Image_Only()
        # model = Model_Tabular_Only()
        # model = MLP_Encoder_Image_Only ()
        # model = CNN_Encoder_Image_Only()
        # model = ResNet50_Dino_Supervised_Image_Only_contrastive()
        # model = ResNet50_Dino_Supervised_Single()

        model = model.to(device)

        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

        print("{:<60} | {:<10}".format("Parameter", "Trainable"))
        print("-" * 75)
        for name, param in model.named_parameters():
            print(f"{name:<60} | {param.requires_grad}")

        
        # for idx, layer in enumerate(model.feature_extractor[6][3:]):
        #     print(f"\nSubmodule {idx + 3} of feature_extractor[6]:")
        #     print(layer)
        #     print("-" * 50)

        num_epochs = 200

        # num of IDs
        # num_negatives = 35 #22 #35   #92 #30
        # num_positives = 22 #35 #22   #18
        # pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
        pos_weight = torch.tensor([1.4], dtype=torch.float32).to(device)
        # print('pos weight:', pos_weight.shape)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = None

        # filters the parameters, keeping only those for which requires_grad=True
        params = filter(lambda p: p.requires_grad, model.parameters())
        lr = 1e-5
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5) #weight_decay=0

        step_size = 20
        gamma = 0.9
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) # step_size=50, gamma=0.6
        # scheduler = None
        tt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        lrs = set([group['lr'] for group in optimizer.param_groups])
        log_tag = "_".join([f"lr{idx}={lr}" for idx, lr in enumerate(sorted(lrs))])
        writer = SummaryWriter(log_dir=f'./runs/my_experiment/{tt}_{log_tag}')

        # Train and validate
        checkpoint, checkpoint_last = train_validate_supervised(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                        device, num_epochs, writer)
        writer.close()
        
        checkpoint['model_summary'] = str(model)
        checkpoint["learning_rate"] = lr
        checkpoint["scheduler_step_size"] = step_size
        checkpoint["scheduler_gamma"] = gamma
        checkpoint["pos_weight"] = pos_weight.item()
        # model_save_path = os.path.join('./saved_models_final_final', f'best_model_fold{fold_idx + 1}_run{run_idx + 1}_seed{seed}_{tt}_{log_tag}.pth')
        model_save_path = os.path.join('./saved_models_final_final', f'best_model_fold{fold_idx + 1}_run{run_idx + 1}_seed{seed}_{tt}_{log_tag}.pth')
        torch.save(checkpoint, model_save_path) 

        last_model_save_path = os.path.join('./saved_models_final_final', f'last_model_fold{fold_idx + 1}_run{run_idx + 1}_seed{seed}_{tt}_{log_tag}.pth')
        torch.save(checkpoint_last, last_model_save_path)

# %%
