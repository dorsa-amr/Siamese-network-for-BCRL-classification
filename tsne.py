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
# from model import ResNet50_Dino_Supervised_ImageTabular, ResNet50_Dino_Supervised_Image_Only, Model_Tabular_Only
from model import ResNet50_Dino
from model import DinoV2_Supervised
from model import twopairs_ResNet50_Dino_Supervised
from model import MLP_Encoder_Image_Only, CNN_Encoder_Image_Only
from loss import ContrastiveLoss
from loss import ContrastiveLoss_weighted
from loss import FocalLoss2
from pytorch_metric_learning import losses
# from dataset import LymphedemaPairDataset, LymphedemaPairDataset_3frames
# from dataset import LymphedemaPairDataset2
# from dataset import SingleLymphedemaDataset
# from dataset import TwopairsLymphedemaPairDataset
from collections import defaultdict
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import vit_small_patch16_224
import random
from torchsummary import summary
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

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

def my_crop(image, output_size=(224,224)):
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

def val_crop(image, output_size=(224,224)):
    # Crop the dots on the right side of the image
    image = TF.crop(image, 0, 0, image.shape[-2] - 5, image.shape[-1] - 8)
    image = TF.resize(image, output_size, interpolation=TF.InterpolationMode.BILINEAR)
    return image

def val_flip_horizontally(image):
    image = torch.flip(image, dims=[-1])
    return image

def val_augmentations(img1, img2):
    # img1, img2 = val_crop(img1),val_crop(img2)
    # img2 = val_flip_horizontally(img2)
    return img1, img2

# def my_augmentations(img):
#     img = my_crop(img)
#     img = my_rotate(img)
#     img = flip_horizontally(img)
#     img = my_random_noise(img)
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
    transforms.Resize((224,224)),  # Doubling the size (224, 224)
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

#%% Dataset   
class LymphedemaPairDataset_3frames(Dataset):
    def __init__(self, parent_folder, participant_ids, info_file, transform=None, augment=None):
        self.parent_folder = parent_folder
        self.participant_ids = participant_ids
        self.transform = transform
        self.augment = augment
        self.info_file = info_file
        self.image_pairs = []
        self.participant_data = self._load_tabular_data()
        self._load_data()
        
    def _load_tabular_data(self):
        """Loads perometry, Total nodes removed, and BMI values from the Excel file."""
        df = pd.read_excel(self.info_file, sheet_name="Sheet1")  # Adjust sheet name if needed

        def replace_nan_with_median(row, medians):
            if pd.isna(row["Total nodes removed"]):
                median_value = medians.get(row["Label2"], 0)
                print(f"Replacing NaN for ID {row['ID']} (Label {row['Label2']}) with median: {median_value}")
                return median_value
            return row["Total nodes removed"]

        # Compute the median values for each label
        medians = df.groupby("Label2")["Total nodes removed"].median().to_dict()
        df["Total nodes removed"] = df.apply(lambda row: replace_nan_with_median(row, medians), axis=1)


        # Convert the required columns into a dictionary
        return {
            str(row["ID"]): {
                "perometry": row["Perometer difference volume calibrated"], #row.get("...", 0),
                "nodes_removed": row["Total nodes removed"],
                "BMI": row["BMI"],
                "Time_surgery": row["Time after surgery"]
            }
            for _, row in df.iterrows()
        }

    
    def _load_data(self): 
        for subfolder in ['no_lymphedema', 'lymphedema']:
            folder_path = os.path.join(self.parent_folder, subfolder)
            if not os.path.isdir(folder_path):
                continue

            participant_images = {pid: {'L2': [], 'R2': [], 'L4': [], 'R4': []} for pid in self.participant_ids}

            for img_name in os.listdir(folder_path):
                if not img_name.endswith('.jpeg'):
                    continue

                participant_id = img_name[:5]
                region = img_name[6:8]  # L2, R2, L4, R4

                if participant_id in participant_images and region in participant_images[participant_id]:
                    image_path = os.path.join(folder_path, img_name)
                    participant_images[participant_id][region].append(image_path)

            for pid, regions in participant_images.items():
                label = 1 if subfolder == 'lymphedema' else 0

                for landmark in ['2', '4']:
                    left_key = f"L{landmark}"
                    right_key = f"R{landmark}"

                    left_images = regions[left_key]
                    right_images = regions[right_key]

                    if left_images and right_images:
                        for l_img in left_images:
                            for r_img in right_images:
                                self.image_pairs.append((l_img, r_img, label))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.image_pairs[idx]
        
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
         
        participant_id = img1_path.split('/')[-1][:5] 

         # Get perometry, nodes removed, and BMI values
        participant_info = self.participant_data.get(participant_id, {"perometry": 0, "nodes_removed": 0, "BMI": 0, "nodes_removed": 0})  # Default to 0 if missing

        # Convert each value to a separate tensor
        x_tab = torch.tensor([
        participant_info["perometry"],
        participant_info["nodes_removed"],
        participant_info["BMI"],
        participant_info["Time_surgery"]
        ], dtype=torch.float32)

        if self.augment:
            img1, img2 = self.augment(img1, img2)
        
        return img1, img2, x_tab, torch.tensor(label, dtype=torch.float32), os.path.basename(img1_path), os.path.basename(img2_path), participant_id
        # return img1, img2, torch.tensor(label, dtype=torch.float32), os.path.basename(img1_path), os.path.basename(img2_path), participant_id

# %%
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
            # logits = model(x_tab)

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
            combined_names = [
                f"{l.replace('.jpeg', '')}_{r.replace('.jpeg', '').split('_', 1)[1]}"
                for l, r in zip(img1_names, img2_names)
            ]
            all_img_names.extend(combined_names)

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

            # image1, image2, image3, image4, labels, img1_path, img2_path, img3_path, img4_path, *_ = batch
            # image1, image2, image3, image4, labels = image1.to(device), image2.to(device), image3.to(device), image4.to(device), labels.to(device).float().unsqueeze(1) 

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
            # logits = model(x_tab)
            # logits, embeds = model(image1,image2, image3, image4)
            
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
            
        tqdm.write((
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
            f"Train F1 Score: {train_f1:.4f}, Val F1 Score: {val_f1:.4f}, "
            # f"Train F1 Score threshold: {best_threshold_acc:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}"
        ))

        # # Check for early stopping
        # early_stopping(val_loss)
        # if epoch >= 150:
        #     if early_stopping.early_stop:
        #         print(f"Stopping training at epoch {epoch+1}")
        #         break  # Stop training if early stopping condition is met

        # # Reduce learning rate at the end of each epoch
        # decay_factor = 0.99
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] *= decay_factor  # Multiply the learning rate by 0.99

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
    
    return checkpoint

#%% Models
class ResNet50_Dino_Supervised_Image_Only(nn.Module):
    def __init__(self, embedding_dim=64):
        """
        Args:
            embedding_dim (int): Dimension of the image embedding.
            tab_embedding_dim (int): Dimension of the tabular embedding, i.e. the output of the positional encoding.
                                     Must be an even number.
        """
        super(ResNet50_Dino_Supervised_Image_Only, self).__init__()

        # Load pre-trained DINO ResNet-50
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
        # Remove classification head (fc layer)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

        # Get the feature dimension after removing the classification head
        feature_dim = 2048  # For DINO ResNet-50, it's 2048

        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            # nn.Dropout(0.2),  # Added on June 10th
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.Dropout(0.2),  # nn.Dropout(0.5), 0.1  #Changed on June 17th
            nn.LeakyReLU(0.05)
        )

        # self.contrastiveOnImage = nn.Sequential(
        #     nn.Linear(embedding_dim, 32), 
        #     nn.LeakyReLU(0.05),
        #     nn.Linear(32, embedding_dim),
        # )

        self.classifier = nn.Linear(embedding_dim, 1)
       
        
        self.freeze_layers()
        

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in list(self.feature_extractor.parameters())[-6:]: #6#7 changed on June 17th #set this to 75 for contrastive weights
            param.requires_grad = True
        
        # for name, param in self.feature_extractor.named_parameters():
        #     if "layer4" in name or "bn" in name:
        #         param.requires_grad = True

        # Ensure embedding layers are trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True


            
    def forward(self, x1, x2): 
        
        # features1 = F.normalize(self.feature_extractor(x1).flatten(1), p=2, dim=1) # Extract features
        # features2 = F.normalize(self.feature_extractor(x2).flatten(1), p=2, dim=1)

        features1 = self.feature_extractor(x1).flatten(1) # Extract features
        features2 = self.feature_extractor(x2).flatten(1)
        # print('FEATURES 1 SHAPE', features1.shape)

        features = torch.abs(features1 - features2)  # Absolute difference
        
        # Generate image-based embedding.
        img_embedding = self.embedding_layer(features)
        # contrastive_features = self.contrastiveOnImage(img_embedding)

        # Compute the classification output
        # norm_embedding = F.normalize(combined_embedding, p=2, dim=1) 
        logit = self.classifier(img_embedding)
    
        return logit.squeeze(1), img_embedding #, contrastive_features #, combined_embedding #logit
    
class ResNet50_Dino_Supervised_ImageTabular(nn.Module):
    def __init__(self, embedding_dim=64, tab_embedding_dim=32):
        """
        Args:
            embedding_dim (int): Dimension of the image embedding.
            tab_embedding_dim (int): Dimension of the tabular embedding, i.e. the output of the positional encoding.
                                     Must be an even number.
        """
        super(ResNet50_Dino_Supervised_ImageTabular, self).__init__()

        # Load pre-trained DINO ResNet-50
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
        # feature_dim = resnet50.fc.in_features if hasattr(resnet50, 'fc') else 2048

        # # Modify first convolutional layer for single-channel input (grayscale)
        # resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head (fc layer)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

        # Get the feature dimension after removing the classification head
        feature_dim = 2048  # For DINO ResNet-50, it's 2048

        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Dropout(0.2),  # Added on June 10th
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.Dropout(0.5),  # nn.Dropout(0.1) #0.5 #Changed on June 17th
            nn.LeakyReLU(0.05)
        )

        # self.tab_embedding_layer = nn.Sequential(
        #     nn.Linear(tab_embedding_dim*2, tab_embedding_dim), 
        #     nn.LeakyReLU(0.05),
        #     nn.Dropout(0.5),
        #     nn.Linear(tab_embedding_dim, tab_embedding_dim),
        #     nn.LeakyReLU(0.05) 
        # )

        tab_dim = 4
        self.tab_embedding_layer = nn.Sequential(
            nn.Linear(tab_dim, 16), 
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(16, tab_embedding_dim),
            nn.LeakyReLU(0.05) 
        )

        # # Instead of a learnable MLP for the tabular data, we use the fixed sinusoidal encoding function.
        # # The positional_scalar_encoding function will produce a tensor of shape (batch, tab_embedding_dim).
        # self.tab_embedding_dim = tab_embedding_dim  # Ensure this is even
        # Combined classifier: image embedding + tabular embedding
        total_embedding_dim = embedding_dim + tab_embedding_dim

        # self.contrastiveOnImage = nn.Sequential(
        #     nn.Linear(embedding_dim, 32), 
        #     nn.LeakyReLU(0.05),
        #     nn.Linear(32, embedding_dim),
        # )

        # self.contrastiveOnImage = nn.Linear(embedding_dim, embedding_dim)

        self.classifier = nn.Linear(total_embedding_dim, 1)  # For binary classification
        # self.classifier = nn.Linear(tab_embedding_dim, 1)  # For binary classification
        # self.classifier = nn.Linear(embedding_dim, 1)
       
        
        self.freeze_layers()
        

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in list(self.feature_extractor.parameters())[-6:]: #7 changed on June 17th
            param.requires_grad = True
        
        # for name, param in self.feature_extractor.named_parameters():
        #     if "layer4" in name or "bn" in name:
        #         param.requires_grad = True

        # Ensure embedding layers are trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

            
    def forward(self, x1, x2, x_tab): 
        
        # features1 = F.normalize(self.feature_extractor(x1).flatten(1), p=2, dim=1) # Extract features
        # features2 = F.normalize(self.feature_extractor(x2).flatten(1), p=2, dim=1)

        features1 = self.feature_extractor(x1).flatten(1) # Extract features
        features2 = self.feature_extractor(x2).flatten(1)
        # print('FEATURES 1 SHAPE', features1.shape)

        features = torch.abs(features1 - features2)  # Absolute difference
        
        # Generate image-based embedding.
        img_embedding = self.embedding_layer(features)
        # contrastive_features = self.contrastiveOnImage(img_embedding)

        # Convert the scalar tabular data into a vector using our positional encoding function.
        # tab_encoded = positional_scalar_encoding(x_tab, self.tab_embedding_dim*2) #self.tab_embedding_dim
        tab_embedding = self.tab_embedding_layer(x_tab)
        
        # Concatenate the image and tabular embeddings.
        combined_embedding = torch.cat((img_embedding, tab_embedding), dim=1)
        
        # Compute the classification output
        # norm_embedding = F.normalize(combined_embedding, p=2, dim=1) 
        logit = self.classifier(combined_embedding)
        # logit = self.classifier(img_embedding)
        # logit = self.classifier(tab_embedding)
    
        return logit.squeeze(1), combined_embedding  #, contrastive_features #, #logit

#%% Data preparation
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
# val_folds = [val_ids_f1]


checkpoint_paths = [
    './saved_models_final_final/July16th_3frames_severity1class0_posweight1.1_image&tabular_tabularnotnormalized_imagesize448*448/tsne/best-run-fold1/best_model1_run1_seed42_2025-07-15_21-54-14_lr0=1e-05.pth',
    './saved_models_final_final/July16th_3frames_severity1class0_posweight1.1_image&tabular_tabularnotnormalized_imagesize448*448/tsne/best-run-fold2/best_model2_run1_seed47_2025-07-16_03-12-08_lr0=1e-05.pth',
    './saved_models_final_final/July16th_3frames_severity1class0_posweight1.1_image&tabular_tabularnotnormalized_imagesize448*448/tsne/best-run-fold3/best_model3_run1_seed52_2025-07-16_08-25-24_lr0=1e-05.pth',
]

model = ResNet50_Dino_Supervised_ImageTabular().to(device)

# To collect feature vectors and labels from all folds
all_features = []
all_labels = []


for fold_idx, (ckpt_path, val_fold_ids) in enumerate(zip(checkpoint_paths, val_folds)):
    print(f"\nLoading model from Fold {fold_idx + 1}...")
    checkpoint = torch.load(ckpt_path)

    # Define train and validation IDs for this fold
    excluded_base_ids = set(id.replace('_REV', '') for id in val_fold_ids + test_ids)
    train_ids = [pid for pid in participant_ids if pid.replace('_REV', '') not in excluded_base_ids]
    # train_ids = [pid for pid in participant_ids if pid not in val_fold_ids + test_ids]
    print(train_ids)

    tabular_file = "./dataset/Id_info.xlsx"

    train_dataset = LymphedemaPairDataset_3frames(root_dir, train_ids, tabular_file, transform=transform, augment=my_augmentations)
    val_dataset = LymphedemaPairDataset_3frames(root_dir, val_ids_f1, tabular_file, transform=transform, augment=val_augmentations)

    print(f"Train dataset pairs: {len(train_dataset)}")
    print(f"Validation dataset pairs: {len(val_dataset)}")

    # data_visualization(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)


    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fold_features = []
    fold_labels = []

    with torch.no_grad():
        for batch in val_loader:
            image1, image2, x_tab, labels, *_ = batch
            image1, image2, x_tab = [x.to(device) for x in (image1, image2, x_tab)]
            labels = labels.to(device).float()

            _, output_features = model(image1, image2, x_tab)
            fold_features.append(output_features.cpu())
            fold_labels.append(labels.cpu())

    all_features.append(torch.cat(fold_features, dim=0))
    all_labels.append(torch.cat(fold_labels, dim=0))

# Flatten to [N_total, D] and [N_total]
features = torch.cat(all_features, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy().astype(int)

print("Final feature shape:", features.shape)
print("Final label shape:", labels.shape)

# Run t-SNE
# Mapping numeric labels to descriptive class names
label_map = {
    0: "0 (No Lymphedema)",
    1: "1 (Lymphedema)"
}
label_names = np.vectorize(label_map.get)(labels)

perplexities = [10] #, 20, 30, 40, 50, 60, 70, 80]

for perp in perplexities:
    print(f"\nRunning t-SNE with perplexity = {perp}")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=label_names,
        palette="deep",
        legend="full"
    )
    # plt.title(f"t-SNE (perplexity={perp})")
    plt.xlabel("t-SNE Dim 1", fontsize=18)
    plt.ylabel("t-SNE Dim 2", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(title="Class", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(f"tsne_perplexity_{perp}.png", dpi=300)
    plt.show()

# # Run 3D t-SNE
# from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plot
# from matplotlib.colors import ListedColormap

# tsne = TSNE(n_components=3, perplexity=30, random_state=42)
# features_3d = tsne.fit_transform(features)

# # Prepare colormap
# unique_labels = np.unique(labels)
# cmap = ListedColormap(plt.cm.tab10.colors[:len(unique_labels)])
# colors = [cmap(i) for i in labels]  # Assign consistent color per class

# # 3D Plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     features_3d[:, 0],
#     features_3d[:, 1],
#     features_3d[:, 2],
#     c=colors,
#     s=40,
#     alpha=0.5
# )

# ax.set_xlabel('t-SNE Dim 1')
# ax.set_ylabel('t-SNE Dim 2')
# ax.set_zlabel('t-SNE Dim 3')
# ax.set_title("3D t-SNE of Multimodal Features (All Folds)")

# # Build legend using same colors
# legend_handles = [
#     plt.Line2D([0], [0], marker='o', color='w',
#                markerfacecolor=cmap(i), label=f'Class {i}', markersize=10)
#     for i in unique_labels
# ]

# ax.legend(handles=legend_handles, title='Class', loc='upper right')
# plt.tight_layout()
# plt.savefig("tsne_3d_color_fixed.png", dpi=300)
# plt.show()


        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # # Run t-SNE
        # tsne = TSNE(n_components=2, random_state=42)
        # features_2d = tsne.fit_transform(features)  # features = your (34, 96) array

        # # Plot
        # plt.figure(figsize=(8, 6))
        # sns.scatterplot(
        #     x=features_2d[:, 0],
        #     y=features_2d[:, 1],
        #     hue=labels1,               # (34,)
        #     palette= 'deep',   #'bright',
        #     legend='full'
        # )
        # plt.xlabel("t-SNE Dimension 1", fontsize=12)
        # plt.ylabel("t-SNE Dimension 2", fontsize=12)
        # plt.title("t-SNE Visualization of Validation Features", fontsize=14)
        # plt.legend(title='Class Label')
        # plt.tight_layout()
        # plt.show()


        # for perp in [10, 20, 30, 40, 50, 60, 70, 80]:
        #     tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        #     features_2d = tsne.fit_transform(features)
            
        #     plt.figure()
        #     sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels1)
        #     plt.xlabel("t-SNE Dimension 1", fontsize=12)
        #     plt.ylabel("t-SNE Dimension 2", fontsize=12)
        #     plt.title(f"t-SNE (perplexity = {perp})")
        #     plt.legend(title='Class Label')
        #     # plt.tight_layout()
        #     # plt.show()

        #     save_name = f"tsne_perplexity_{perp}.png"
        #     plt.savefig(save_name, dpi=300)
        #     print(f"Saved {save_name}")
        #     plt.close()

#%% averaging ALL 5 RUNS

# all_features = []  # Will be a list of tensors of shape [N_samples, feature_dim]
# labels1 = None     # Will be set once

# for path in checkpoint_files:
#     print(f"\nLoading checkpoint: {path}")
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     features_this_ckpt = []
#     labels_this_ckpt = []

#     with torch.no_grad(): 
#         for batch in val_loader:
#             image1, image2, x_tab , labels, *_ = batch
#             image1, image2, x_tab, labels = [x.to(device) for x in (image1, image2, x_tab, labels)]
#             labels = labels.float()

#             logits, output_features = model(image1, image2, x_tab)
#             features_this_ckpt.append(output_features.cpu())
#             labels_this_ckpt.append(labels.cpu())

#     features_this_ckpt = torch.cat(features_this_ckpt, dim=0)  # [N, D]
#     all_features.append(features_this_ckpt)

#     if labels1 is None:
#         labels1 = torch.cat(labels_this_ckpt, dim=0)  # [N]

# # # Check shapes
# # for i, f in enumerate(all_features):
# #     print(f"Checkpoint {i+1}: feature shape = {f.shape}")

# # Stack and average
# all_features_stacked = torch.stack(all_features, dim=0)  # [num_ckpts, N, D]
# features_avg = torch.mean(all_features_stacked, dim=0)   # [N, D]

# print("Averaged features shape:", features_avg.shape)
# print("Labels shape:", labels1.shape)


# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Run t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# features_2d = tsne.fit_transform(features_avg.numpy())  # features = your (34, 96) array

# # Plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     x=features_2d[:, 0],
#     y=features_2d[:, 1],
#     hue=labels1,               # (34,)
#     palette= 'deep',   #'bright',
#     legend='full'
# )

# plt.xlabel("t-SNE Dimension 1", fontsize=12)
# plt.ylabel("t-SNE Dimension 2", fontsize=12)
# plt.title("t-SNE Visualization of Validation Features", fontsize=14)
# plt.legend(title='Class Label')
# plt.tight_layout()
# plt.show()