"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='hie_per', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator', 'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, i)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, n=0):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        elif subarch == 'hie_per':
            netD = HiePerDiscriminator(opt, max(1, 3-n))
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [
            [
                nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False),
            ]
        ]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [
                [
                    norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                    nn.LeakyReLU(0.2, False),
                ]
            ]

        sequence += [
            [
                nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)
            ]
        ]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


# Defines the Hierarchical Perceptual discriminator with the specified arguments.
class HiePerDiscriminator(BaseNetwork):
    def __init__(self, opt, n_addition_layer=3):
        super().__init__()
        assert n_addition_layer > 0
        self.opt = opt
        max_dim = 512
        self.n_add_layer = n_addition_layer
        self.backboneNet = Vgg16backbone(block_num=min(5, 2+n_addition_layer), requires_grad=False)

        activation = nn.LeakyReLU(0.2, False)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        sequence = [
            nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1),
            activation,
        ]
        model_1 = nn.Sequential(*sequence)
        sequence = [
            norm_layer(nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1)),
            activation,
        ]
        model_2 = nn.Sequential(*sequence)
        sequence = [
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1)),
            activation,
        ]
        model_3 = nn.Sequential(*sequence)
        self.common = nn.ModuleList([model_1, model_2, model_3])

        for i in range(n_addition_layer):
            nf = max_dim if i > 0 else max_dim // 2
            sequence = [
                norm_layer(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)),
                activation,
            ]
            model = nn.Sequential(*sequence)
            setattr(self, 'pre' + str(i + 1), model)

            nf = 512 if i == 0 else 1024
            sequence = [
                norm_layer(nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1)),
                activation,
            ]
            model = nn.Sequential(*sequence)
            setattr(self, 'gcb' + str(i + 1), model)

        for i in range(n_addition_layer):
            if i != n_addition_layer - 1:
                nf = 512
                sequence = [
                    InceptionBlock(nf, 512, stride=2, norm_layer=norm_layer, activation=activation),
                ]
                model = nn.Sequential(*sequence)
                setattr(self, 'conv' + str(i + 1) + '_line', model)

            nf = 512
            sequence = [
                norm_layer(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)),
                activation,
                nn.Conv2d(nf, 1, kernel_size=5, stride=1, padding=2),
                nn.Upsample(scale_factor=2**i),
            ]
            model = nn.Sequential(*sequence)
            setattr(self, 'conv' + str(i + 1) + '_out', model)

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def re_normalize(self, input, mean=[0.485, 0.456, 0.406], var=[0.229, 0.224, 0.225]):
        input = (input + 1.0) * 0.5 # [0.0, 1.0]
        r = (input[:, 0:1, :, :] - mean[0]) / var[0]
        g = (input[:, 1:2, :, :] - mean[1]) / var[1]
        b = (input[:, 2:3, :, :] - mean[2]) / var[2]
        return torch.cat((r, g, b), 1)


    def forward(self, input):
        image = self.re_normalize(input[:, -self.opt.output_nc:, :, :])
        results = [input]
        out = []
        for layer in self.common:
            feat_pool = layer(results[-1])
            results.append(feat_pool)
        backbone_fea = self.backboneNet(image)

        for i in range(self.n_add_layer):
            backbone_fea_this = backbone_fea[2+i]
            pre = getattr(self, 'pre' + str(i + 1))
            gcb = getattr(self, 'gcb' + str(i + 1))
            backbone_fea_this = pre(backbone_fea_this)
            fea = torch.cat((results[-1], backbone_fea_this), 1)
            fea = gcb(fea)
            results.append(fea)
            outBlock = getattr(self, 'conv' + str(i + 1) + '_out')
            out_this = outBlock(fea)
            out.append(out_this)
            if i < self.n_add_layer - 1:
                lineBlock = getattr(self, 'conv' + str(i + 1) + '_line')
                feat_pool = lineBlock(fea)
                results.append(feat_pool)
        out = torch.cat(out, 1)
        results.append(out)
        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class Vgg16backbone(nn.Module):
    def __init__(self, block_num, requires_grad=False):
        super(Vgg16backbone, self).__init__()
        assert block_num <= 5
        self.block_num = block_num
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        if block_num >= 1:
            self.slice1 = nn.Sequential()
            for x in range(5):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 2:
            self.slice2 = nn.Sequential()
            for x in range(5, 10):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 3:
            self.slice3 = nn.Sequential()
            for x in range(10, 17):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 4:
            self.slice4 = nn.Sequential()
            for x in range(17, 24):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 5:
            self.slice5 = nn.Sequential()
            for x in range(24, 31):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = []
        pool = x
        for i in range(self.block_num):
            model = getattr(self, 'slice' + str(i + 1))
            pool = model(pool)
            out.append(pool)
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_layer, activation):
        super(InceptionBlock, self).__init__()
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride),
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            activation,
        )
        self.conv3_branch = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            activation,
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)),
            activation,
        )
        self.conv5_branch = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            activation,
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation,
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)),
            activation,
        )

    def forward(self, x):
        pool_b = self.pool_branch(x)
        conv3_b = self.conv3_branch(x)
        conv5_b = self.conv5_branch(x)
        out = pool_b + conv3_b + conv5_b
        return out