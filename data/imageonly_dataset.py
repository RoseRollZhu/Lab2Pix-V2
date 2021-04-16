from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
from data.image_folder import make_dataset

class ImageonlyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        image_paths = make_dataset(opt.results_dir, recursive=False, read_cache=True)

        util.natural_sort(image_paths)

        self.image_paths = image_paths

        self.dataset_size = len(image_paths)

    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        return {'image':image_tensor}

    def __len__(self):
        return self.dataset_size
