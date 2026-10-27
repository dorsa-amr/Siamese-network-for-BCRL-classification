"""Models of the paper (Sec. III-C and IV):
    UnimodalA          Siamese network, ultrasound images only
    Multimodal         Siamese network, ultrasound images + tabular clinical data (Fig. 3)
    UnimodalB          tabular clinical data only
    UnimodalA_CL       Unimodal-A with a projection head for supervised contrastive (SupCon) learning
    UnimodalNoSiamese  single-arm network (one image at a time)
"""
import torch
from torch import nn

FEATURE_DIM = 2048   # output size of the ResNet-50 encoder
TAB_DIM = 4          # volume difference, number of dissected nodes, time since surgery, BMI


def dino_resnet50_encoder():
    """ResNet-50 pre-trained with DINO on ImageNet-1K (facebookresearch/dino, main branch) without its
    classification head; returns 2048-D features."""
    resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    return nn.Sequential(*list(resnet50.children())[:-1])


def freeze_encoder(encoder, n_trainable=6):
    """Partial fine-tuning (Sec. III-C1): freezes the encoder except its last `n_trainable` parameter tensors.
    The last 6 tensors are the last two convolutional layers of ResNet-50 and their batch-norm layers."""
    for param in encoder.parameters():
        param.requires_grad = False
    if n_trainable > 0:
        for param in list(encoder.parameters())[-n_trainable:]:
            param.requires_grad = True


def image_mlp(embedding_dim=64, dropout=0.2):
    """Two fully connected layers with Leaky ReLU activations and dropout: 2048-D -> 64-D image embedding."""
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 256),
        nn.LeakyReLU(0.05),
        nn.Linear(256, embedding_dim),
        nn.Dropout(dropout),
        nn.LeakyReLU(0.05),
    )


def tabular_mlp(tab_embedding_dim=32):
    """Two fully connected layers with dropout 0.5: 4-D tabular features -> 32-D tabular embedding."""
    return nn.Sequential(
        nn.Linear(TAB_DIM, 16),
        nn.LeakyReLU(0.05),
        nn.Dropout(0.5),
        nn.Linear(16, tab_embedding_dim),
        nn.LeakyReLU(0.05),
    )


class UnimodalA(nn.Module):
    """Siamese network: shared encoder for the left- and right-arm images, absolute difference of the two
    feature vectors, MLP and classifier."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.feature_extractor = dino_resnet50_encoder()
        self.embedding_layer = image_mlp(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
        freeze_encoder(self.feature_extractor)

    def image_embedding(self, x1, x2):
        features1 = self.feature_extractor(x1).flatten(1)
        features2 = self.feature_extractor(x2).flatten(1)
        return self.embedding_layer(torch.abs(features1 - features2))

    def forward(self, x1, x2):
        return self.classifier(self.image_embedding(x1, x2)).squeeze(1)


class Multimodal(UnimodalA):
    """Unimodal-A image embedding (64-D) concatenated with a tabular embedding (32-D), then a classifier (Fig. 3)."""

    def __init__(self, embedding_dim=64, tab_embedding_dim=32):
        super().__init__(embedding_dim)
        self.tab_embedding_layer = tabular_mlp(tab_embedding_dim)
        self.classifier = nn.Linear(embedding_dim + tab_embedding_dim, 1)

    def combined_embedding(self, x1, x2, x_tab):
        return torch.cat((self.image_embedding(x1, x2), self.tab_embedding_layer(x_tab)), dim=1)

    def forward(self, x1, x2, x_tab):
        return self.classifier(self.combined_embedding(x1, x2, x_tab)).squeeze(1)


class UnimodalB(nn.Module):
    """Tabular data only: 4-D features -> 32-D embedding -> classifier."""

    def __init__(self, tab_embedding_dim=32):
        super().__init__()
        self.tab_embedding_layer = tabular_mlp(tab_embedding_dim)
        self.classifier = nn.Linear(tab_embedding_dim, 1)

    def forward(self, x_tab):
        return self.classifier(self.tab_embedding_layer(x_tab)).squeeze(1)


class UnimodalA_CL(UnimodalA):
    """Unimodal-A with a projection head on the image embedding for the SupCon loss (Sec. III-C2).

    pretrain=True:  stage 1, the encoder (last 75 parameter tensors and the batch-norm layers of all residual
                    blocks), the MLP and the projection head are trained with the SupCon loss.
    pretrain=False: stage 2, the encoder (initialized with the stage-1 weights) and the projection head are
                    frozen; the MLP (dropout increased to 0.5 and an extra dropout layer of 0.5) and the
                    classifier are fine-tuned with the weighted BCE loss.
    """

    def __init__(self, embedding_dim=64, pretrain=False):
        super().__init__(embedding_dim)
        self.contrastiveOnImage = nn.Sequential(  # projection head
            nn.Linear(embedding_dim, 32),
            nn.LeakyReLU(0.05),
            nn.Linear(32, embedding_dim),
        )
        if pretrain:
            freeze_encoder(self.feature_extractor, n_trainable=75)
            for name, param in self.feature_extractor.named_parameters():
                if 'bn' in name:
                    param.requires_grad = True
        else:
            self.embedding_layer = nn.Sequential(
                nn.Linear(FEATURE_DIM, 256),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.05),
                nn.Linear(256, embedding_dim),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.05),
            )
            freeze_encoder(self.feature_extractor, n_trainable=0)
            for param in self.contrastiveOnImage.parameters():
                param.requires_grad = False

    def forward(self, x1, x2):
        img_embedding = self.image_embedding(x1, x2)
        return self.classifier(img_embedding).squeeze(1), self.contrastiveOnImage(img_embedding)


class UnimodalNoSiamese(nn.Module):
    """Same encoder, MLP and classifier as Unimodal-A, applied to a single image."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.feature_extractor = dino_resnet50_encoder()
        self.embedding_layer = image_mlp(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
        freeze_encoder(self.feature_extractor)

    def forward(self, x):
        return self.classifier(self.embedding_layer(self.feature_extractor(x).flatten(1))).squeeze(1)
