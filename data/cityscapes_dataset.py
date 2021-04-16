"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import torch


class CityscapesDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=35)
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(batchSize=16)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt, image_only=False):
        if not image_only:
            root = opt.dataroot
            phase = 'val' if opt.phase == 'test' else 'train'

            label_dir = os.path.join(root, 'gtFine', phase)
            label_paths_all = make_dataset(label_dir, recursive=True)
            label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

            image_dir = os.path.join(root, 'leftImg8bit', phase)
            image_paths = make_dataset(image_dir, recursive=True)

            if not opt.no_instance:
                instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
            else:
                instance_paths = []

            sys.path.insert(0, os.path.join(root, 'scripts/helpers'))
            labels = __import__('labels')
            self.id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs

            return label_paths, image_paths, instance_paths
        else:
            root = opt.results_dir

            image_paths = make_dataset(root, recursive=True)

            return image_paths


    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])


    # In cityscapes, transfrom category number from 35 to 19 when eval
    def postprocess(self, input_dict):
        label = input_dict['label']
        label_19c = torch.zeros_like(label)
        for k,v in self.id2trainId.items():
            label_19c[label == k] = v
        input_dict['label_19c'] = label_19c
        return input_dict
