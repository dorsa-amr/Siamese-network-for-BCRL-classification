# Point-of-Care Ultrasound in Diagnosis of Breast Cancer-Related Lymphedema

Code for the paper **"Point-of-Care Ultrasound in Diagnosis of Breast Cancer-Related Lymphedema: A Multimodal Database and Deep Learning Approach"** by Dorsa Ameri, Ali K. Z. Tehrani, Benoit Freiche, Anna Towers and Hassan Rivaz.

The method classifies breast cancer-related lymphedema (BCRL) from bilateral point-of-care ultrasound (POCUS) B-mode images of the arms. It uses a Siamese network with a shared DINO-pretrained ResNet-50 encoder and the absolute difference of the left- and right-arm features. It can optionally fuse clinical tabular data (arm volume difference, number of dissected lymph nodes, time since surgery and BMI).

| Method in the paper | Input | Script |
|---|---|---|
| Unimodal-A | left + right arm images (Siamese) | `train.py` (default) |
| Multimodal | images + tabular data | `train.py` (`[Multimodal]` lines) |
| Unimodal-B | tabular data only | `train_tabular_only.py` |
| Unimodal-A+CL | images, encoder pre-trained with supervised contrastive (SupCon) loss | `train.py` (`[Unimodal-A+CL pre-train]` lines), then `train_contrastive_finetune.py` |
| Unimodal-no-Siamese | one image (single arm) | `train.py` (`[Unimodal-no-Siamese]` lines) |

## Repository structure

```
common.py                       paths, cross-validation folds, pre-processing, augmentation, metrics
dataset.py                      datasets (image pairs, single images, tabular data)
model.py                        the five models of the paper
train.py                        training of Unimodal-A / Multimodal / Unimodal-no-Siamese / SupCon pre-training
train_contrastive_finetune.py   Unimodal-A+CL fine-tuning on the SupCon pre-trained encoder
train_tabular_only.py           Unimodal-B
evaluate.py                     averaged and ensemble metrics (Table IV, Supplementary Tables I and III)
gradcam.py                      Grad-CAM maps of Unimodal-A (Supplementary Fig. 1)
tsne.py                         t-SNE of the Multimodal embeddings (Fig. 4)
stats/statistical_tests.py      Unimodal-B vs. Multimodal statistical tests (Supplementary Table II)
preprocessing/crop_and_check_flipping.py   cropping and orientation check of the exported frames
```

## Installation

```bash
pip install -r requirements.txt
```

The DINO ResNet-50 weights are downloaded automatically through `torch.hub` (`facebookresearch/dino:main`) the first time a model is built. A GPU is strongly recommended.

## Data

The scripts expect the following layout (paths are set in `common.py`):

```
dataset/
├── Id_info_uploaded.xlsx                      clinical data, one row per subject (sheet "Sheet1")
└── folders/3_frame/severity1_as_class0/
    ├── no_lymphedema/   US009_L2_a.jpeg, US009_L2_b.jpeg, US009_L2_c.jpeg, US009_R2_a.jpeg, ...
    └── lymphedema/      US019_L2_a.jpeg, ...
```

- Image names are `<subject ID>_<L|R><landmark>_<frame>.jpeg`. Only landmarks 2 and 4 are used, with three frames (`a`, `b`, `c`) per landmark and arm. The images are the 581×446 B-mode frames produced by `preprocessing/crop_and_check_flipping.py`.
- Class folders: `lymphedema` holds subjects with severity > 1. `no_lymphedema` holds subjects without lymphedema or with very mild lymphedema (severity ≤ 1).
- Columns used from `Id_info_uploaded.xlsx`: `ID`, `Label` (0/1, as above), `Perometer difference volume calibrated`, `Total nodes removed`, `BMI`, `Time after surgery` (years) and `Affected side` (`L`/`R`; only used by Unimodal-no-Siamese).

## Training

All methods use three-fold cross-validation with subject-level splits (the folds are listed in `common.py`). Each method trains five instances per fold with seeds 42–46, runs for 200 epochs, and keeps the checkpoint with the lowest validation loss. Checkpoints are saved to `checkpoints/<experiment>/Fold<k>/` and TensorBoard logs to `runs/`.

### Switching between methods in `train.py`

`train.py` trains several methods. Every switchable line ends with a tag, and every switchable block starts with a tag line:

```python
logits = model(image1, image2)                          # [Unimodal-A]
# logits = model(image1, image2, x_tab)                 # [Multimodal]
```

To train a method, make active (uncomment) every line or block whose tag contains that method. Comment out every other tagged line or block. Untagged lines are shared by all methods. The script trains Unimodal-A by default.

| Method | Steps |
|---|---|
| Unimodal-A | `python train.py` |
| Multimodal | activate the `[Multimodal]` lines in `train.py`, then `python train.py` |
| Unimodal-no-Siamese | activate the `[Unimodal-no-Siamese]` lines in `train.py`, then `python train.py` |
| Unimodal-A+CL | activate the `[Unimodal-A+CL pre-train]` lines in `train.py` and run `python train.py` (stage 1, saved to `checkpoints/unimodal_a_cl_pretrain/`); then `python train_contrastive_finetune.py` (stage 2) |
| Unimodal-B | `python train_tabular_only.py` |

### Hyperparameters (Table III)

All models use Adam with weight decay 1e-5, a step learning-rate scheduler, a weighted BCE loss with positive-class weight p<sub>c</sub>, batch size 12 and 200 epochs. The SupCon pre-training stage of Unimodal-A+CL uses batch size 60 and one run per fold.

| Method | p<sub>c</sub> | base lr | γ | step s |
|---|---|---|---|---|
| Unimodal-A | 1.4 | 1e-5 | 0.9 | 20 |
| Multimodal | 1.1 | 1e-5 | 0.9 | 20 |
| Unimodal-B | 1.1 | 1e-4 | 0.6 | 50 |
| Unimodal-A+CL | 1.4 | 1e-5 | 0.9 | 20 |

Partial fine-tuning: all ResNet-50 layers except the last two convolutional layers are frozen. The SupCon loss uses temperature τ = 0.1.

## Evaluation

```bash
python evaluate.py checkpoints/multimodal                        # any experiment folder with Fold1-Fold3
python evaluate.py checkpoints/multimodal --oof-dir outputs/oof/multimodal   # also save the out-of-fold predictions
```

The script prints two strategies, each as mean ± SD and 95% CI over the three folds:

- **Averaged:** each run's metrics at its best validation loss, averaged over the five runs.
- **Ensemble:** the five runs' probabilities averaged per sample, thresholded at 0.5. This strategy also reports AUPRC.

## Figures and statistics

```bash
python gradcam.py               # Grad-CAM of Unimodal-A on the validation samples -> outputs/gradcam/Fold<k>/
python tsne.py                  # t-SNE of the Multimodal embeddings -> outputs/tsne_perplexity_10.png
python stats/statistical_tests.py
```

## Citation

If you use this code, please cite:

```
D. Ameri, A. K. Z. Tehrani, B. Freiche, A. Towers, and H. Rivaz, "Point-of-Care Ultrasound in Diagnosis of
Breast Cancer-Related Lymphedema: A Multimodal Database and Deep Learning Approach."
```

## License

This project is released under the MIT License (see `LICENSE`).
