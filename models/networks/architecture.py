"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import DGNorm


# ResNet block that uses DG-Norm.
class DGNormResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = DGNorm(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = DGNorm(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = DGNorm(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, global_seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, global_seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, global_seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# Global Encoder
class GlobalEncoder(nn.Module):
    def __init__(self, fin, fout, nf, num_downsample):
        super().__init__()

        self.model_down = []
        self.model_up = nn.ModuleList()
        self.num_downsample = num_downsample

        f_in = fin
        f_out = nf
        for i in range(num_downsample):
            if i == 0:
                self.model_down += [
                    spectral_norm(nn.Conv2d(f_in, f_out, kernel_size=3, stride=1, padding=1)),
                    nn.LeakyReLU(2e-1),
                ]
            else:
                self.model_down += [
                    spectral_norm(nn.Conv2d(f_in, f_out, kernel_size=3, stride=2, padding=1)),
                    nn.LeakyReLU(2e-1),
                ]
            f_in = f_out
            f_out = min(f_out * 2, 512)
        self.model_down = nn.Sequential(*self.model_down)

        for i in range(num_downsample):
            model_up = nn.Sequential(
                nn.Upsample(scale_factor=2**(num_downsample-i-1)),
                nn.Conv2d(f_in, fout, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.model_up.append(model_up)
            
    def forward(self, x):
        out = []
        down_x = self.model_down(x)
        for i in range(self.num_downsample):
            up_x = self.model_up[i](down_x)
            out.append(up_x)
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
