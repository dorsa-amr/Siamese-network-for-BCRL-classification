import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torchvision.transforms as transforms
from model import ResNet50_Dino_Supervised_ImageTabular, ResNet50_Dino_Supervised_Image_Only
from torch.utils.data import DataLoader
from dataset import LymphedemaPairDataset, LymphedemaPairDataset_3frames 


def run_gradcam_ensemble(models_folder, val_loader, save_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    os.makedirs(save_dir, exist_ok=True)

    # Find all model paths
    model_paths = sorted(glob.glob(os.path.join(models_folder, "*.pth")))
    print(f"Found {len(model_paths)} models for ensemble.")

    # Store results for CSV
    results = []

    # # Target layer
    # dummy_model = ResNet50_Dino_Supervised_Image_Only()
    # target_layer = dummy_model.feature_extractor[7][2].conv3  # just to reference the layer structure

    def get_cam(model, x1, x2, target_layer):
        gradients, activations = [], []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Register hooks
        f_hook = target_layer.register_forward_hook(forward_hook)
        b_hook = target_layer.register_backward_hook(backward_hook)

        # Prepare inputs
        x1 = x1.clone().detach().to(device).requires_grad_(True)
        x2 = x2.clone().detach().to(device).requires_grad_(True)

        model.eval().to(device)
        logit = model(x1, x2)  # shape [1]

        model.zero_grad()
        logit.sum().backward()  # just use the scalar logit for backward

        # Compute CAM
        grads = gradients[0]
        acts = activations[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Remove hooks
        f_hook.remove()
        b_hook.remove()
        return cam

    # Iterate over validation batches
    for co, batch in enumerate(tqdm(val_loader, desc="Grad-CAM Ensemble")):
        image1, image2, labels, img1_name, img2_name, *_ = batch
        labels = labels.float()
        batch_size = image1.size(0)

        for i in range(batch_size):
            x1 = image1[i:i+1]
            x2 = image2[i:i+1]
            label = labels[i].item()

            cam1_sum = 0
            cam2_sum = 0
            prob_sum = 0

            # Compute Grad-CAM and probability for all models
            for model_path in model_paths:
                model = ResNet50_Dino_Supervised_Image_Only()
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                 # Identify target layer AFTER loading weights
                target_layer = model.feature_extractor[7][2].conv3

                cam1 = get_cam(model, x1, x2, target_layer)
                cam2 = get_cam(model, x2, x1, target_layer)
                cam1_sum += cam1
                cam2_sum += cam2

                with torch.no_grad():
                    logits = model(x1.to(device), x2.to(device))
                    probs = torch.sigmoid(logits)
                    prob_sum += probs.item()

            # Average cams and probability
            cam1_avg = cam1_sum / len(model_paths)
            cam2_avg = cam2_sum / len(model_paths)
            avg_prob = prob_sum / len(model_paths)

            # Normalize cams jointly
            global_min = min(cam1_avg.min(), cam2_avg.min())
            global_max = max(cam1_avg.max(), cam2_avg.max())
            cam1_avg = (cam1_avg - global_min) / (global_max - global_min)
            cam2_avg = (cam2_avg - global_min) / (global_max - global_min)

            # Prepare image for overlay
            def prep_img(x):
                img = x.squeeze().permute(1, 2, 0).cpu().numpy()
                return (img - img.min()) / (img.max() - img.min())

            img1_np = prep_img(x1)
            img2_np = prep_img(x2)

            def overlay(img_np, cam):
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                return 0.4 * heatmap + 0.6 * img_np

            overlay1 = overlay(img1_np, cam1_avg)
            overlay2 = overlay(img2_np, cam2_avg)

            # Plot
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(overlay1)
            axs[0].set_title(os.path.basename(img1_name[i]))
            axs[0].axis('off')
            axs[1].imshow(overlay2)
            axs[1].set_title(os.path.basename(img2_name[i]))
            axs[1].axis('off')

            fig.suptitle(f"Avg Prob: {avg_prob:.2f} | GT: {label}")
            plt.tight_layout()

            # Add colorbar
            im2 = axs[1].imshow(overlay2)
            cbar = fig.colorbar(im2, ax=axs, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Activation')

            # Save
            clean_img1 = os.path.basename(img1_name[i]).replace(" ", "_").replace(".jpeg","")
            clean_img2 = os.path.basename(img2_name[i]).replace(" ", "_").replace(".jpeg","")
            outname = f"{clean_img1}_{clean_img2}_avgprob{avg_prob:.2f}.png"
            plt.savefig(os.path.join(save_dir, outname), bbox_inches='tight')
            plt.close()

            # --- Save info for CSV ---
            results.append({
            'img1_name': img1_name,
            'img2_name': img2_name,
            'GT': int(label),
            'avg_prob': avg_prob
        })


    # --- Create CSV sorted by prob ---
    df = pd.DataFrame(results)
    df_class0 = df[df['GT'] == 0].sort_values('avg_prob')
    df_class1 = df[df['GT'] == 1].sort_values('avg_prob')

   # Create two tables side-by-side (same index alignment)
    df_merged = pd.concat([df_class0.reset_index(drop=True), df_class1.reset_index(drop=True)], axis=1)
    df_merged.columns = [
        'img1_name_0', 'img2_name_0', 'GT_0', 'avg_prob_0',
        'img1_name_1', 'img2_name_1', 'GT_1', 'avg_prob_1'
    ]

    csv_path = os.path.join(save_dir, "ensemble_probabilities.csv")
    df_merged.to_csv(csv_path, index=False)
    print(f"\nSaved sorted probabilities to: {csv_path}")


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

def my_rotate(image):
    if np.random.random() > 0.5:
        angle = np.random.randint(-5, 6)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR) 
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


# def my_augmentations(img):
#     img = my_crop(img)
#     img = my_rotate(img)
#     # img = flip_horizontally(img)
#     img = my_random_noise(img)
#     return img


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Doubling the size (224 * 2 = 448)
    transforms.ToTensor(),  # Converts to [0, 1] range
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Duplicate the single channel to 3 channels for not needing to train the first conv layer
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load val_dataset

root_dir = './dataset/folders/3_frame/severity1_as_class0' 

tabular_file = "./dataset/Id_info_uploaded.xlsx"

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

val_fold_ids = val_ids_f2

val_dataset = LymphedemaPairDataset_3frames(root_dir, val_fold_ids, tabular_file, transform=transform, augment=val_augmentations)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


train_ids = [pid for pid in participant_ids if pid not in val_fold_ids]
print(train_ids)
train_dataset = LymphedemaPairDataset_3frames(root_dir, train_ids, tabular_file, transform=transform, augment=my_augmentations)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


models_folder = './saved_models_final_final/July9th_3frames_severity1class0_posweight1.4_image_only_imagesize448*448_batchsizes1&1/new/Fold3'

# Run Grad-CAM
run_gradcam_ensemble(models_folder, train_loader, save_dir='./gradcam_outputs')
