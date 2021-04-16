import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from trainers.pix2pix_trainer import Pix2PixTrainer
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse()

assert len(opt.gpu_ids) == 1

os.environ['MASTER_ADDR'] = opt.master_address
os.environ['MASTER_PORT'] = opt.master_port

torch.backends.cudnn.benchmark = True
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

dataloader = data.create_dataloader(opt)

trainer = Pix2PixTrainer(opt)

if local_rank == 0:
    visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
if local_rank == 0:
    web_dir = os.path.join(opt.results_dir, opt.name,
                        '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = trainer.inference(data_i)

    if local_rank == 0:
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([
                ('input_label',       data_i['label'][b]),
                ('synthesized_image', generated[b])
            ])
            visualizer.save_images(webpage, visuals, img_path[b:b+1])

webpage.save()
