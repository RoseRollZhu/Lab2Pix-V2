"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import DGNormResnetBlock as DGNormResnetBlock
from models.networks.architecture import GlobalEncoder as GlobalEncoder


class LabelComprehensiveGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadebatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh, self.num_up_layers = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.fpencoder = GlobalEncoder(opt.semantic_nc, 128, 64, self.num_up_layers)

        self.head_0 = DGNormResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = DGNormResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = DGNormResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = DGNormResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = DGNormResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = DGNormResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = DGNormResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = DGNormResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh, num_up_layers+1

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        encode_seg = self.fpencoder(seg)
        current_scale = -1

        x = self.head_0(x, seg, encode_seg[current_scale])

        x = self.up(x)
        current_scale -= 1
        x = self.G_middle_0(x, seg, encode_seg[current_scale])

        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            current_scale -= 1

        x = self.G_middle_1(x, seg, encode_seg[current_scale])

        x = self.up(x)
        current_scale -= 1
        x = self.up_0(x, seg, encode_seg[current_scale])
        x = self.up(x)
        current_scale -= 1
        x = self.up_1(x, seg, encode_seg[current_scale])
        x = self.up(x)
        current_scale -= 1
        x = self.up_2(x, seg, encode_seg[current_scale])
        x = self.up(x)
        current_scale -= 1
        x = self.up_3(x, seg, encode_seg[current_scale])

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            current_scale -= 1
            x = self.up_4(x, seg, encode_seg[current_scale])

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x
