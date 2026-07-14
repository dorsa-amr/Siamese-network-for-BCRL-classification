import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights, ResNet50_Weights
import numpy as np
# import torchvision
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
import torch.nn.functional as F
# import os
# import matplotlib.pyplot as plt
# from utils import *
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import types
import math
# from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
# import pdb
#from network import Resnet
#from network.mynn import initialize_weights, Norm2d

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=1, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)
        self.soft2 = nn.Sigmoid()

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        #Using Softmax instead of Sigmoid
        #return self.soft2(self.final(out))
        return self.soft(self.final(out))


class UNext_classification(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes=3, input_channels=1, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs): #, num_classes_classification=2
        super().__init__()
        
        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)
        self.soft2 = nn.Sigmoid()

        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        # # Define the classifier layer
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(embed_dims[-1] * (img_size // 16) * (img_size // 16), 128), 
        #     nn.ReLU(),
        #     nn.Linear(128, num_classes_classification)
        # )

    def forward(self, x): #, return_bottleneck, return_segmentation=False #return_classification=False
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        #out = F.relu(self.adaptive_pool(self.ebn1(self.encoder1(x))))
        t1 =  out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        #out = F.relu(self.adaptive_pool(self.ebn2(self.encoder2(out))))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        #out = F.relu(self.adaptive_pool(self.ebn3(self.encoder3(out))))  # Always outputs (batch_size, channels, 1, 1) 
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #size: [4, 256, 7, 7]
        out_bottleneck = torch.mean(out,dim=(-2,-1)) #size: [4, 256]
        #out_bottleneck =  torch.mean(out,dim=1, keepdim=True) #size: [4, 1, 7, 7])
        #out_bottleneck = out
        # print('out_bottleneck: ',torch.mean(out_bottleneck**2,dim=1))
        norm_out_bottleneck = F.normalize(out_bottleneck, p=2, dim=1)  # L2 normalize embeddings
        # print('norm_out_bottleneck: ',torch.mean(norm_out_bottleneck**2,dim=1))
        # if return_bottleneck:
        return out_bottleneck
        # return norm_out_bottleneck
        
        # #if return_bottleneck:
        # #   return bottleneck_output  # For feature extraction or custom uses
        
        # # if return_classification:
        # #     # Pass the bottleneck output through the classifier for classification
        # #     # out_classification = self.classifier(out_bottleneck)
        # #     return  out_classification 

        
        # if return_segmentation:
        #     ### Stage 4

        #     out = F.relu(F.interpolate(self.dbn1(self.decoder1(out_bottleneck)),scale_factor=(2,2),mode ='bilinear'))
            
        #     out = torch.add(out,t4)
        #     _,_,H,W = out.shape
        #     out = out.flatten(2).transpose(1,2)
        #     for i, blk in enumerate(self.dblock1):
        #         out = blk(out, H, W)

        #     ### Stage 3
            
        #     out = self.dnorm3(out)
        #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #     out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        #     out = torch.add(out,t3)
        #     _,_,H,W = out.shape
        #     out = out.flatten(2).transpose(1,2)
            
        #     for i, blk in enumerate(self.dblock2):
        #         out = blk(out, H, W)

        #     out = self.dnorm4(out)
        #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        #     out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        #     out = torch.add(out,t2)
        #     out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        #     out = torch.add(out,t1)
        #     out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        #     #Using Softmax instead of Sigmoid 
        #     #return self.soft2(self.final(out))
        #     out_segmentation = self.soft(self.final(out)) # Using Softmax for segmentation output
        #     return out_segmentation
           
        # # If neither flag is set, return None or raise an exception
        # raise ValueError("Specify either return_classification=True or return_segmentation=True")
    
# class added_fcs(nn.Module):
#     def__init(self, )
#     super().__init__()


class UNext_Wrapper(nn.Module):
    def __init__(self, pretrained_model_path, num_classes=3, input_channels=1, **kwargs):
        super(UNext_Wrapper, self).__init__()
        
        # Initialize the UNext_classification model with desired parameters
        self.model = UNext_classification(num_classes=num_classes, **kwargs)
        
        # Load pre-trained weights
        self.load_pretrained_weights(pretrained_model_path)

    def load_pretrained_weights(self, pretrained_model_path):
        # Load the state dictionary from the pre-trained model
        pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
        
        # Get the current model's parameters
        model_dict = self.model.state_dict()
        
        # Filter out unnecessary keys from the pre-trained dictionary
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # Update the current model's state dictionary with pre-trained weights
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        print("Pre-trained weights loaded successfully.")

    def forward(self, x, return_bottleneck=True, return_segmentation=False):
        # Forward pass through the UNext_classification model
        return self.model(x) #, return_bottleneck=return_bottleneck, return_segmentation=return_segmentation)
    
    # def compute_euclidean_distance(self, input1, input2):
    #     # Obtain the bottleneck features for each input
    #     out_bottleneck1 = self.model(input1, return_bottleneck=True)
    #     out_bottleneck2 = self.model(input2, return_bottleneck=True)
        
    #     euclidean_distance = F.pairwise_distance(out_bottleneck1, out_bottleneck2, p=2)
        
    #     return euclidean_distance
    #     # Usage: 
    #     # distances = model.compute_euclidean_distance(input1, input2)
        
class UNext_S(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)
        
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)


#EOF

class GSCNN(nn.Module):
    '''
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    '''

    def __init__(self, num_classes, trunk=None, criterion=None):
        
        super(GSCNN, self).__init__()
        self.num_classes = num_classes

        wide_resnet = wider_resnet38_a2(classes=5, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        
        wide_resnet = wide_resnet.module
        self.mod1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = F.interpolate
        del wide_resnet

        self.dsn1 = nn.Conv2d(16, 1, 1)
        self.dsn3 = nn.Conv2d(64, 1, 1)
        self.dsn4 = nn.Conv2d(128, 1, 1)
        self.dsn7 = nn.Conv2d(1024, 1, 1)

        self.res1 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d1 = nn.Conv2d(16, 8, 1)
        self.res2 = Resnet.BasicBlock(8, 8, stride=1, downsample=None)
        self.d2 = nn.Conv2d(8, 4, 1)
        self.res3 = Resnet.BasicBlock(4, 4, stride=1, downsample=None)
        self.d3 = nn.Conv2d(4, 2, 1)
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(8, 8)
        self.gate2 = gsc.GatedSpatialConv2d(4, 4)
        self.gate3 = gsc.GatedSpatialConv2d(2, 2)
         
        self.aspp = _AtrousSpatialPyramidPoolingModule(1024, 64,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(32, 12, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(320 + 64, 64, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(64 + 12, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.final_seg)

    def forward(self, inp, gts=None):

        x_size = inp.size() 

        # res 1
        m1 = self.mod1(inp)

        # res 2
        m2 = self.mod2(self.pool2(m1))
        
        # res 3
        m3 = self.mod3(self.pool3(m2))
          
        # res 4-7
        m4 = self.mod4(m3)
        m5 = self.mod5(m4)
        m6 = self.mod6(m5)
        m7 = self.mod7(m6) 

        s3 = F.interpolate(self.dsn3(m3), x_size[2:],
                            mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(m4), x_size[2:],
                            mode='bilinear', align_corners=True)
        s7 = F.interpolate(self.dsn7(m7), x_size[2:],
                            mode='bilinear', align_corners=True)
        
        m1f = F.interpolate(m1, x_size[2:], mode='bilinear', align_corners=True)

        im_arr = inp.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(m1f)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)
        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s7)
        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        # aspp
        x = self.aspp(m7, acts)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = self.interpolate(dec0_up, m2.size()[2:], mode='bilinear',align_corners=True)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final_seg(dec0)  
        seg_out = self.interpolate(dec1, x_size[2:], mode='bilinear')            
       
        return seg_out
    

class ResNet(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super(ResNet, self).__init__()
        

        # Load a pre-trained ResNet model (Updated for PyTorch 2.0+)
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        
         # Modify the first convolutional layer to accept 1 channel instead of 3
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification head
        
        # Add a fully connected layer for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),  # Flatten the ResNet output
            nn.Dropout(0.5),
            nn.Linear(resnet.fc.in_features, 256),  # First FC layer (hidden dimension)
            # resnet.fc.in_features = 512
            nn.LeakyReLU(0.05), 
            nn.Linear(256, 128),  # Second FC layer 
            nn.LeakyReLU(0.05),
            nn.Linear(128, embedding_dim), #final embedding
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output

        # self.freeze_layers()

    # def freeze_layers(self):
    #     # Freeze the entire ResNet model first
    #     c = 0
    #     for param in self.feature_extractor.parameters():
    #         if (c==0): 
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    #         c = c+1
        
    #     for param in self.embedding_layer.parameters():
    #         param.requires_grad = True


    def forward(self, x):
        features = self.feature_extractor(x)   
        # print(features.shape)       # Extract ResNet features
        embedding = self.embedding_layer(features)    # Generate the embedding
        # norm_embedding = F.normalize(embedding, p=2, dim=1) 
    
        logit = self.classifier(embedding)  # Output raw logit
        # return logit  # No sigmoid here (BCEWithLogitsLoss applies it)
        return logit, embedding
        # return embedding


class ResNet_Supervised(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super(ResNet_Supervised, self).__init__()

        # Load a pre-trained ResNet model (Updated for PyTorch 2.0+)
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)


         # Modify the first convolutional layer to accept 1 channel instead of 3
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification head
        
        # Add a fully connected layer for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet.fc.in_features, 256),  # First FC layer (hidden dimension)
            # nn.Linear(2*resnet.fc.in_features, 256),  # First FC layer (hidden dimension)
            # resnet.fc.in_features = 512
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Second FC layer 
            nn.LeakyReLU(0.05)
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output

        # self.freeze_layers()

    # def freeze_layers(self):
    #     # Freeze the entire ResNet model first
    #     c = 0
    #     for param in self.feature_extractor.parameters():
    #         if (c==0): 
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    #         c = c+1
        
    #     for param in self.embedding_layer.parameters():
    #         param.requires_grad = True


    def forward(self, x1,x2):
        features1 = self.feature_extractor(x1).flatten(1)
        features2 = self.feature_extractor(x2).flatten(1)
        # print(features1.shape) 
        # if np.random.random() > 0.5:
        #     features = torch.cat((features1,features2),dim=1) 
        # else:
        #     features = torch.cat((features2,features1),dim=1) 

        features = torch.abs(features1-features2) 
        # print(features.shape)       # Extract ResNet features
        embedding = self.embedding_layer(features)    # Generate the embedding
        # norm_embedding = F.normalize(embedding, p=2, dim=1) 
    
        logit = self.classifier(embedding)  # Output raw logit
        # return logit  # No sigmoid here (BCEWithLogitsLoss applies it)
        # return logit, embedding
        return logit,embedding

class ResNet50_Dino(nn.Module):
    def __init__(self, embedding_dim):
        super(ResNet50_Dino, self).__init__()

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
            nn.Flatten(),  # Flatten the ResNet output
            nn.Dropout(0),
            nn.Linear(feature_dim, 256),  # First FC layer
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.LeakyReLU(0.05)
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output
        # self.freeze_layers()

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        c = 0
        for param in self.feature_extractor.parameters():
                param.requires_grad = False
                c=c+1
        
        c2=0
        for param in self.feature_extractor.parameters():
            if c2>(c-8):
                param.requires_grad = True
            c2=c2+1
        
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.feature_extractor(x)   
        # print(features.shape)       # Extract ResNet features
        embedding = self.embedding_layer(features)    # Generate the embedding
        # norm_embedding = F.normalize(embedding, p=2, dim=1) 
    
        logit = self.classifier(embedding)  # Output raw logit
        # return logit  # No sigmoid here (BCEWithLogitsLoss applies it)
        return logit, embedding


class ResNet50_Dino_SupervisedandContrastive(nn.Module):
    def __init__(self, embedding_dim):
        super(ResNet50_Dino_Supervised, self).__init__()

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
            nn.Dropout(0),
            nn.Linear(feature_dim, 256),  # First FC layer
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.LeakyReLU(0.05)
        )

        self.embedding_contrastive = nn.Sequential(
            nn.Linear(feature_dim, 256),  # First FC layer
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output
        # self.freeze_layers()

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        c = 0
        for param in self.feature_extractor.parameters():
                param.requires_grad = False
                c=c+1
        
        c2=0
        for param in self.feature_extractor.parameters():
            if c2>(c-8):
                param.requires_grad = True
            c2=c2+1
        
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

    def forward(self, x1, x2):

        features1 = self.feature_extractor(x1).flatten(1)  # Extract features
        features2 = self.feature_extractor(x2).flatten(1)

        features = torch.abs(features1 - features2)  # Absolute difference

        embedding = self.embedding_layer(features)  # Generate embedding

        embedding_contrastive = self.embedding_contrastive (features)
    
        logit = self.classifier(embedding)  # Compute classification output

        return logit, embedding_contrastive
    

class ResNet50_Dino_Supervised(nn.Module):
    def __init__(self, embedding_dim): # pretrained=True
        super(ResNet50_Dino_Supervised, self).__init__()

        # Load pre-trained DINO ResNet-50
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        # weights = ResNet50_Weights.DEFAULT if pretrained else None
        # resnet = models.resnet50(weights=weights)

        # feature_dim = resnet50.fc.in_features if hasattr(resnet50, 'fc') else 2048

        # Modify first convolutional layer for single-channel input (grayscale)
        # resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head (fc layer)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

        # Get the feature dimension after removing the classification head
        feature_dim = 2048  # For DINO ResNet-50, it's 2048

        # Fully connected layers for embedding generation
        # self.embedding_layer = nn.Sequential(
        #     nn.Dropout(0),
        #     nn.Linear(feature_dim, 256),  # First FC layer
        #     nn.LeakyReLU(0.05), 
        #     nn.Linear(256, embedding_dim),  # Output embedding
        #     nn.LeakyReLU(0.05)
        # )

        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),  # First FC layer , resnet.fc.in_features
            nn.LeakyReLU(0.05), 
            nn.Dropout(0.5), #0
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.LeakyReLU(0.05)
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output
        self.freeze_layers()

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Unfreeze the last few layers
        for param in list(self.feature_extractor.parameters())[-7:]:
            param.requires_grad = True
        # Ensure embedding layers are trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

    def forward(self, x1, x2):

        features1 = self.feature_extractor(x1).flatten(1)  # Extract features
        features2 = self.feature_extractor(x2).flatten(1)

        features = torch.abs(features1 - features2)  # Absolute difference

        embedding = self.embedding_layer(features)  # Generate embedding
    
        logit = self.classifier(embedding)  # Compute classification output

        return logit, embedding
    

class twopairs_ResNet50_Dino_Supervised(nn.Module):
    def __init__(self, embedding_dim):
        super(twopairs_ResNet50_Dino_Supervised, self).__init__()
        # Load pre-trained DINO ResNet-50
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        # resnet50.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head (fc layer)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        feature_dim = 2048  # For DINO ResNet-50, it's 2048
        
        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(feature_dim*2, 256),  # Adjusted input dimension after concatenation
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.LeakyReLU(0.05)
        )
        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output
        self.freeze_layers()

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Unfreeze the last few layers
        for param in list(self.feature_extractor.parameters())[-7:]:
            param.requires_grad = True
        # Ensure embedding layers are trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

    def forward(self, x1, x2, x3, x4):
        # Extract features for each image
        # f1 = self.feature_extractor(x1,x3).flatten(1)
        # f2 = self.feature_extractor(x2,x4).flatten(1)

        # # Concatenate features: f1 with f3, and f2 with f4
        # features1 = torch.cat((f1, f3), dim=1)
        # features2 = torch.cat((f2, f4), dim=1)

        # Compute absolute difference
        # features = torch.abs(features1 - features2)

        # Generate embedding
        embedding = self.embedding_layer(features)

        # Compute classification output
        logit = self.classifier(embedding)

        return logit, embedding



def positional_scalar_encoding(scalar_feature, embedding_dim):
    """
    Converts a scalar feature into a sinusoidal embedding vector.
    
    Args:
        scalar_feature (torch.Tensor): A tensor of shape (batch_size,) or (batch_size, 1)
                                       containing scalar values.
        embedding_dim (int): The desired dimension of the embedding vector. Must be even.
    
    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing the sinusoidal embedding.
    """
    # Ensure the scalar is a 2D tensor with shape (batch_size, 1)
    scalar_feature = scalar_feature.view(-1, 1)
    
    # Create frequency bands: size (embedding_dim//2,)
    freqs = torch.exp(torch.linspace(0, np.log(80), embedding_dim // 2, device=scalar_feature.device))
    
    # Compute sinusoidal embeddings
    sin_emb = torch.sin(scalar_feature * freqs)
    cos_emb = torch.cos(scalar_feature * freqs)
    
    # Concatenate along the last dimension to get shape (batch_size, embedding_dim)
    sinusoidal_embedding = torch.cat([sin_emb, cos_emb], dim=-1)
    return sinusoidal_embedding



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

    # def freeze_layers(model):
    #     for name, param in model.named_parameters():
    #         # Freeze early ResNet layers (feature extractor blocks 0–5)
    #         if name.startswith("feature_extractor.0") or \
    #             name.startswith("feature_extractor.1") or \
    #             name.startswith("feature_extractor.4") or \
    #             name.startswith("feature_extractor.5"):
    #             param.requires_grad = False
            
    #         # # Optionally freeze or unfreeze parts of 6.0 to 6.2 (early mid-level)
    #         # elif name.startswith("feature_extractor.6.0") or \
    #         #     name.startswith("feature_extractor.6.1") or \
    #         #     name.startswith("feature_extractor.6.2") or \
    #         #     name.startswith("feature_extractor.6.3") or \
    #         #     name.startswith("feature_extractor.6.4") or \
    #         #     name.startswith("feature_extractor.6.5") or \
    #         #     name.startswith("feature_extractor.7.0") or \
    #         #     name.startswith("feature_extractor.7.1"):
    #         #     param.requires_grad = False

    #         # Unfreeze higher layers: 6.3 and above (more domain-specific)
    #         elif name.startswith("feature_extractor.7.2.conv2") or \
    #             name.startswith("feature_extractor.7.2.bn2") or \
    #             name.startswith("feature_extractor.7.2.conv3") or \
    #             name.startswith("feature_extractor.7.2.bn3"):
    #             param.requires_grad = True

    #         # Unfreeze all custom layers (your additions)
    #         elif name.startswith("embedding_layer") or \
    #             name.startswith("tab_embedding_layer") or \
    #             name.startswith("classifier") or \
    #             name.startswith("contrastiveOnImage"):
    #             param.requires_grad = True

    #         else:
    #             # For safety, freeze everything else by default
    #             param.requires_grad = False


            
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
    
        return logit.squeeze(1) #, contrastive_features #, combined_embedding #logit

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
    
        return logit.squeeze(1) #, contrastive_features #, combined_embedding #logit
    
class ResNet50_Dino_Supervised_Image_Only_contrastive(nn.Module):
    def __init__(self, embedding_dim=64):
        """
        Args:
            embedding_dim (int): Dimension of the image embedding.
            tab_embedding_dim (int): Dimension of the tabular embedding, i.e. the output of the positional encoding.
                                     Must be an even number.
        """
        super(ResNet50_Dino_Supervised_Image_Only_contrastive, self).__init__()

        # Load pre-trained DINO ResNet-50
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
        # Remove classification head (fc layer)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

        # Get the feature dimension after removing the classification head
        feature_dim = 2048  # For DINO ResNet-50, it's 2048

        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Dropout(0.5), #0.2  # Added on June 10th
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.Dropout(0.5),  # nn.Dropout(0.5), 0.1  #Changed on June 17th
            nn.LeakyReLU(0.05)
        )

        self.contrastiveOnImage = nn.Sequential(
            nn.Linear(embedding_dim, 32), 
            nn.LeakyReLU(0.05),
            nn.Linear(32, embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, 1)
       
        
        self.freeze_layers()
        

    def freeze_layers(self):
        # Freeze the entire ResNet model first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # # Unfreeze the last few layers
        # for param in list(self.feature_extractor.parameters())[-75:]: #6#7 changed on June 17th #Set this to 75 for contrastive weights
        #     param.requires_grad = True
        
        # for name, param in self.feature_extractor.named_parameters():
        #     if "layer4" in name or "bn" in name:
        #         param.requires_grad = True

        # Ensure embedding layers are/are not trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True #False #True

        for param in self.contrastiveOnImage.parameters():
            param.requires_grad = False #False
            
    def forward(self, x1, x2): 
        
        # features1 = F.normalize(self.feature_extractor(x1).flatten(1), p=2, dim=1) # Extract features
        # features2 = F.normalize(self.feature_extractor(x2).flatten(1), p=2, dim=1)

        features1 = self.feature_extractor(x1).flatten(1) # Extract features
        features2 = self.feature_extractor(x2).flatten(1)
        # print('FEATURES 1 SHAPE', features1.shape)

        features = torch.abs(features1 - features2)  # Absolute difference
        
        # Generate image-based embedding.
        img_embedding = self.embedding_layer(features)
        contrastive_features = self.contrastiveOnImage(img_embedding)

        # Compute the classification output
        # norm_embedding = F.normalize(combined_embedding, p=2, dim=1) 
        logit = self.classifier(img_embedding)
    
        return logit.squeeze(1), contrastive_features #, combined_embedding #logit
    
class ResNet50_Dino_Supervised_Single(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ResNet50_Dino_Supervised_Single, self).__init__()

        # Load pre-trained DINO ResNet-50
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        # Remove classification head (fc layer)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

        feature_dim = 2048  # DINO ResNet-50 feature dimension

        # Fully connected embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.05),
            nn.Linear(256, embedding_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.05)
        )

        # Final classifier
        self.classifier = nn.Linear(embedding_dim, 1)

        # Freeze backbone layers except the last few
        self.freeze_layers()

    def freeze_layers(self):
        # Freeze most of ResNet
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in list(self.feature_extractor.parameters())[-6:]:
            param.requires_grad = True

        # Keep embedding and classifier trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Single image forward pass
        features = self.feature_extractor(x).flatten(1)
        img_embedding = self.embedding_layer(features)
        logit = self.classifier(img_embedding)
        return logit.squeeze(1)
    
class Model_Tabular_Only(nn.Module):
    def __init__(self, tab_embedding_dim=32):
        """
        Args:
            embedding_dim (int): Dimension of the image embedding.
            tab_embedding_dim (int): Dimension of the tabular embedding, i.e. the output of the positional encoding.
                                     Must be an even number.
        """
        super(Model_Tabular_Only, self).__init__()

        # Load pre-trained DINO ResNet-50
        # resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
        # # feature_dim = resnet50.fc.in_features if hasattr(resnet50, 'fc') else 2048

        # # # Modify first convolutional layer for single-channel input (grayscale)
        # # resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # # Remove classification head (fc layer)
        # self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

        # # Get the feature dimension after removing the classification head
        # feature_dim = 2048  # For DINO ResNet-50, it's 2048


        tab_dim = 4
        self.tab_embedding_layer = nn.Sequential(
            nn.Linear(tab_dim, 16), 
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(16, tab_embedding_dim),
            nn.LeakyReLU(0.05) 
        )


        self.classifier = nn.Linear(tab_embedding_dim, 1)  # For binary classification
    
        
    #     self.freeze_layers()
        

    # def freeze_layers(self):
    #     # Freeze the entire ResNet model first
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = False

    #     # Unfreeze the last few layers
    #     for param in list(self.feature_extractor.parameters())[-6:]: #7 changed on June 17th
    #         param.requires_grad = True
        
    #     # for name, param in self.feature_extractor.named_parameters():
    #     #     if "layer4" in name or "bn" in name:
    #     #         param.requires_grad = True

    #     # Ensure embedding layers are trainable
    #     for param in self.embedding_layer.parameters():
    #         param.requires_grad = True

            
    def forward(self, x_tab):       

        tab_embedding = self.tab_embedding_layer(x_tab)
        
        logit = self.classifier(tab_embedding)
    
        return logit.squeeze(1)


class MobileNetV2_Supervised_Tabular(nn.Module):
    def __init__(self, embedding_dim=64, tab_embedding_dim=32):
        """
        Args:
            embedding_dim (int): Dimension of the image embedding.
            tab_embedding_dim (int): Dimension of the tabular embedding.
        """
        super(MobileNetV2_Supervised_Tabular, self).__init__()

        # Load pre-trained MobileNetV2
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        
        # Modify first convolutional layer for single-channel input (grayscale)
        # If your images are grayscale, uncomment the following lines
        # mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Remove classification head
        self.feature_extractor = mobilenet.features  # Extract only convolutional layers

        # Get the feature dimension after removing the classification head
        feature_dim = 1280  # MobileNetV2's feature dimension

        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.05), 
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
            nn.LeakyReLU(0.05)
        )

        tab_dim = 4
        self.tab_embedding_layer = nn.Sequential(
            nn.Linear(tab_dim, 16), 
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(16, tab_embedding_dim),
            nn.LeakyReLU(0.05) 
        )

        # Combined classifier: image embedding + tabular embedding
        total_embedding_dim = embedding_dim + tab_embedding_dim
        self.classifier = nn.Linear(total_embedding_dim, 1)  # Binary classification
        self.contrastiveOnImage = nn.Linear(embedding_dim, embedding_dim)

        self.freeze_layers()

    def freeze_layers(self):
        # Freeze all layers first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers properly
        for layer in list(self.feature_extractor.children())[-1:]:  # Last 5 *blocks*, not params
            for param in layer.parameters():
                param.requires_grad = True
        
        # Ensure embedding layers are trainable
        for param in self.embedding_layer.parameters():
            param.requires_grad = True


    def forward(self, x1, x2, x_tab):
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)

        # Global Average Pooling to match the feature dimensions
        features1 = features1.mean([2, 3])
        features2 = features2.mean([2, 3])

        features = torch.abs(features1 - features2)  # Absolute difference
        
        # Generate image-based embedding
        img_embedding = self.embedding_layer(features)
        contrastive_features = self.contrastiveOnImage(img_embedding)

        # Tabular feature embedding
        tab_embedding = self.tab_embedding_layer(x_tab)

        # Concatenate the image and tabular embeddings
        combined_embedding = torch.cat((img_embedding, tab_embedding), dim=1)

        # Compute the classification output
        logit = self.classifier(combined_embedding)

        return logit.squeeze(1), contrastive_features



class SimpleCNN_Dino_Supervised_Tabular(nn.Module):
    def __init__(self, embedding_dim=64, tab_embedding_dim=32):
        """
        Args:
            embedding_dim (int): Dimension of the image embedding.
            tab_embedding_dim (int): Dimension of the tabular embedding.
        """
        super(SimpleCNN_Dino_Supervised_Tabular, self).__init__()

        # Simple CNN feature extractor for 224x224 images
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 224, 224)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Output: (256, 14, 14)
        )

        # Compute feature dimension after CNN (for 224×224 input)
        self.feature_dim = 256 * 14 * 14  # = 50176

        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.LeakyReLU(0.05)
        )

        tab_dim = 4
        self.tab_embedding_layer = nn.Sequential(
            nn.Linear(tab_dim, 16),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(16, tab_embedding_dim),
            nn.LeakyReLU(0.05)
        )

        # Combined classifier: image embedding + tabular embedding
        total_embedding_dim = embedding_dim + tab_embedding_dim
        self.classifier = nn.Linear(total_embedding_dim, 1)  # Binary classification
        self.contrastiveOnImage = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x1, x2, x_tab):
        # Extract image features
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)

        # Flatten features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # Compute absolute difference
        features = torch.abs(features1 - features2)

        # Generate image embedding
        img_embedding = self.embedding_layer(features)
        contrastive_features = self.contrastiveOnImage(img_embedding)

        # Convert tabular data into an embedding
        tab_embedding = self.tab_embedding_layer(x_tab)

        # Concatenate the image and tabular embeddings
        combined_embedding = torch.cat((img_embedding, tab_embedding), dim=1)

        # Compute classification output
        logit = self.classifier(combined_embedding)

        return logit.squeeze(1), contrastive_features



class ViT_S8_Dino_Supervised(nn.Module):
    def __init__(self, embedding_dim):
        super(ViT_S8_Dino_Supervised, self).__init__()

        # Load pre-trained DINO ViT-S/8
        self.feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

        # Feature dimension for ViT-S/8
        feature_dim = self.feature_extractor.embed_dim  # Typically 384 for ViT-S/8

        # Fully connected layers for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(feature_dim, 256),  # First FC layer
            nn.LeakyReLU(0.05),
            nn.Linear(256, embedding_dim),  # Output embedding
            nn.LeakyReLU(0.05)
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)  # Single neuron output
        self.freeze_layers()

    def freeze_layers(self):
        # Freeze the entire ViT model first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze the last transformer block
        for param in self.feature_extractor.blocks[-1].parameters():
            param.requires_grad = True

        # Unfreeze the embedding and classifier layers
        for param in self.embedding_layer.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x1, x2):
        # Extract features
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)

        # Absolute difference
        features = torch.abs(features1 - features2)

        # Generate embedding
        embedding = self.embedding_layer(features)

        # Compute classification output
        logit = self.classifier(embedding)

        return logit, embedding


class DinoV2_Supervised(nn.Module):
    def __init__(self, embedding_dim):
        super(DinoV2_Supervised, self).__init__()

        # Load pre-trained DINOv2 ViT-S/14
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')

        # Freeze most layers (e.g., except the last transformer blocks)
        # self.freeze_layers()

        # Feature dimension for ViT-S/14 is 384
        feature_dim = 384

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.05),
            nn.Linear(256, embedding_dim),
            nn.LeakyReLU(0.05)
        )

        # Classification layer for binary classification
        self.classifier = nn.Linear(embedding_dim, 1)
        self.freeze_layers()

    def freeze_layers(self):
        # Freeze all layers first
        for param in self.dinov2.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 transformer blocks (if available)
        # Typically, ViT blocks are in self.dinov2.blocks or self.dinov2.transformer.layers
        try:
            for param in list(self.dinov2.blocks[-2:].parameters()):
                param.requires_grad = True
        except AttributeError:
            print("Layer structure may differ. Adjust manually.")

    def forward(self, x1, x2):
        # Extract features from DINOv2
        features1 = self.dinov2.backbone.forward_features(x1)['x_norm_clstoken']
        features2 = self.dinov2.backbone.forward_features(x2)['x_norm_clstoken']

        # Compute absolute difference between feature embeddings
        features = torch.abs(features1 - features2)

        # Generate embedding
        embedding = self.embedding_layer(features)

        # Classification output
        logit = self.classifier(embedding)

        return logit, embedding


class MobileNetBinaryClassifier(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super(MobileNetBinaryClassifier, self).__init__()

        # Load MobileNetV2 (pretrained)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)

        # Modify first layer for 1-channel input (grayscale images)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.feature_extractor = mobilenet.features  # Extract feature layers

        # Embedding + Classification head
        self.embedding_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool features
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(embedding_dim, 1)  # Binary classification

    def forward(self, x):
        features = self.feature_extractor(x)
        embedding = self.embedding_layer(features)
        output = self.classifier(embedding)
        return output, embedding

class MobileNet_Supervised(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super(MobileNet_Supervised, self).__init__()

        # Load MobileNetV2 (pretrained)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)

        # Modify first layer for 1-channel input (grayscale images)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.feature_extractor = mobilenet.features  # Extract feature layers

        # Embedding + Classification head
        self.embedding_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool features
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(embedding_dim, 1)  # Binary classification

    def forward(self, x1,x2):
        features1 = self.feature_extractor(x1).flatten(1)
        features2 = self.feature_extractor(x2).flatten(1)
        # print(features1.shape)  

        features = torch.abs(features1-features2) 
        # print(features.shape)       # Extract ResNet features
        embedding = self.embedding_layer(features)    # Generate the embedding
        # norm_embedding = F.normalize(embedding, p=2, dim=1) 
        logit = self.classifier(embedding)  # Output raw logit
        # return logit  # No sigmoid here (BCEWithLogitsLoss applies it)
        return logit,embedding
    
class DenseNet(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(DenseNet, self).__init__()
        
        # Load a pre-trained model
        densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

        # Modify the first convolutional layer to accept 1 channel instead of 3
        densenet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.feature_extractor = nn.Sequential(*list(densenet.children())[:-1])  # Remove the classification head
        
        # Add a fully connected layer for embedding generation
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),  # Flatten the ResNet output
            nn.Linear(densenet.fc.in_features, embedding_dim),  # Project to embedding_dim
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)          # Extract ResNet features
        embedding = self.embedding_layer(features)    # Generate the embedding
        return embedding

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = None  # Placeholder, will be defined dynamically later
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Dynamically compute feature size
        x = x.view(x.size(0), -1)  # Flatten
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 512).to(x.device)  # Initialize fc1 dynamically

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class SwinV2Model(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinV2Model, self).__init__()

        # Load the SwinV2 model
        self.model = timm.create_model(
            'swin_large_patch4_window7_224_in22k',  # Example model name
            pretrained=pretrained,
            num_classes=0  # Removes the classification head
        )
        
        for param in self.model.parameters():param.requires_grad=False

        # Output dimension of the last feature layer in SwinV2
        self.in_dim = self.model.num_features

        # Fully connected layer to reduce feature size to 256
        self.fc1 = nn.Linear(self.in_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        # self.fc3 = nn.Linear(512, 256)


        # Dropout for regularization
        # self.dropout = nn.Dropout(p=0.3)

        # Initialize weights for the custom layers
        # self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Duplicate single channel to create a 3-channel image
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Pass input through the SwinV2 model
        x = self.model(x)
        
        # x = F.relu(x)
        # # Reduce feature size to 256
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = F.relu(x)
        
        # x = self.fc3(x)

        return x
    


class MLP_Encoder_Image_Only(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=512):
        """
        Args:
            embedding_dim (int): Output dimension of the image embedding.
            hidden_dim (int): Hidden dimension for the MLP encoder.
        """
        super(MLP_Encoder_Image_Only, self).__init__()

        # Simple MLP encoder on flattened images
        self.feature_extractor = nn.Sequential(
            nn.Linear(3 * 224 * 224, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),  # final feature_dim = 256
            nn.ReLU()
        )

        feature_dim = 256

        # Fully connected layers for embedding generation (same as your previous)
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.05), 
            nn.Linear(256, embedding_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.05)
        )

        self.classifier = nn.Linear(embedding_dim, 1)

        self.freeze_layers()

    def freeze_layers(self):
        # No freezing needed in this MLP, but kept for consistency
        pass

    def forward(self, x1, x2):
        # Flatten images
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)

        # Encode using MLP
        features1 = self.feature_extractor(x1_flat)
        features2 = self.feature_extractor(x2_flat)

        # Absolute difference for pairwise comparison
        features = torch.abs(features1 - features2)

        # Generate image-based embedding
        img_embedding = self.embedding_layer(features)

        # Classification output
        logit = self.classifier(img_embedding)

        return logit.squeeze(1)


class CNN_Encoder_Image_Only(nn.Module):
    def __init__(self, embedding_dim=64):
        """
        Args:
            embedding_dim (int): Output dimension of the image embedding.
        """
        super(CNN_Encoder_Image_Only, self).__init__()

        # Simple CNN encoder
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224x224 -> 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112 -> 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56 -> 28x28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 28x28 -> 1x1
        )

        feature_dim = 256  # After global pooling

        # # Fully connected layers for embedding generation
        # self.embedding_layer = nn.Sequential(
        #     nn.Linear(feature_dim, 256),
        #     nn.LeakyReLU(0.05), 
        #     nn.Linear(256, embedding_dim),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.05)
        # )

        # self.classifier = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x1, x2):
        # Extract features
        features1 = self.feature_extractor(x1).flatten(1)  # [B, 256, 1, 1] -> [B, 256]
        features2 = self.feature_extractor(x2).flatten(1)

        # Absolute difference for pairwise comparison
        features = torch.abs(features1 - features2)

        # # Generate image-based embedding
        # img_embedding = self.embedding_layer(features)

        # Classification output
        # logit = self.classifier(img_embedding)
        logit = self.classifier(features)

        return logit.squeeze(1)


# class ResNet50_Dino_Supervised_Image_Only_contrastive(nn.Module):
#     def __init__(self, embedding_dim=64):
#         """
#         Args:
#             embedding_dim (int): Dimension of the image embedding.
#             tab_embedding_dim (int): Dimension of the tabular embedding, i.e. the output of the positional encoding.
#                                      Must be an even number.
#         """
#         super(ResNet50_Dino_Supervised_Image_Only_contrastive, self).__init__()

#         # Load pre-trained DINO ResNet-50
#         resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
#         # Remove classification head (fc layer)
#         self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

#         # Get the feature dimension after removing the classification head
#         feature_dim = 2048  # For DINO ResNet-50, it's 2048

#         # Fully connected layers for embedding generation
#         self.embedding_layer = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             # nn.Dropout(0.2),  # Added on June 10th
#             nn.LeakyReLU(0.05), 
#             nn.Linear(256, embedding_dim),  # Output embedding
#             nn.Dropout(0.2),  # nn.Dropout(0.5), 0.1  #Changed on June 17th
#             nn.LeakyReLU(0.05)
#         )

#         self.contrastiveOnImage = nn.Sequential(
#             nn.Linear(embedding_dim, 32), 
#             nn.LeakyReLU(0.05),
#             nn.Linear(32, embedding_dim),
#         )

#         self.classifier = nn.Linear(embedding_dim, 1)
       
        
#         self.freeze_layers()
        

#     def freeze_layers(self):
#         # Freeze the entire ResNet model first
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False

#         # # Unfreeze the last few layers
#         # for param in list(self.feature_extractor.parameters())[-75:]: #6#7 changed on June 17th #Set this to 75 for contrastive weights
#         #     param.requires_grad = True
        
#         # for name, param in self.feature_extractor.named_parameters():
#         #     if "layer4" in name or "bn" in name:
#         #         param.requires_grad = True

#         # Ensure embedding layers are/are not trainable
#         for param in self.embedding_layer.parameters():
#             param.requires_grad = False #True


            
#     def forward(self, x1, x2): 
        
#         # features1 = F.normalize(self.feature_extractor(x1).flatten(1), p=2, dim=1) # Extract features
#         # features2 = F.normalize(self.feature_extractor(x2).flatten(1), p=2, dim=1)

#         features1 = self.feature_extractor(x1).flatten(1) # Extract features
#         features2 = self.feature_extractor(x2).flatten(1)
#         # print('FEATURES 1 SHAPE', features1.shape)

#         features = torch.abs(features1 - features2)  # Absolute difference
        
#         # Generate image-based embedding.
#         img_embedding = self.embedding_layer(features)
#         contrastive_features = self.contrastiveOnImage(img_embedding)

#         # Compute the classification output
#         # norm_embedding = F.normalize(combined_embedding, p=2, dim=1) 
#         logit = self.classifier(img_embedding)
    
#         return logit.squeeze(1), contrastive_features #, combined_embedding #logit
    