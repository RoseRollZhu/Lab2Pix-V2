import os
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch

# parse options
opt = TrainOptions().parse()
os.environ['MASTER_ADDR'] = opt.master_address
os.environ['MASTER_PORT'] = opt.master_port

torch.backends.cudnn.benchmark = True
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

# print options to help debugging
if local_rank == 0:
    print(' '.join(sys.argv))

# load the dataset
dataloader, sampler = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
if local_rank == 0:
    visualizer = Visualizer(opt)

if local_rank == 0:
    print('Start training...')
for epoch in iter_counter.training_epochs():
    sampler.set_epoch(epoch)
    if local_rank == 0:
        iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        if local_rank == 0:
            iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if local_rank == 0 and iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if local_rank == 0 and iter_counter.needs_displaying():
            visuals = OrderedDict([
                ('input_label',       data_i['label']),
                ('synthesized_image', trainer.get_latest_generated()),
                ('real_image',        data_i['image'])
            ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if local_rank == 0 and iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    if local_rank == 0:
        iter_counter.record_epoch_end()

    if local_rank == 0 and \
        (epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

if local_rank == 0:
    print('Training was successfully finished.')
