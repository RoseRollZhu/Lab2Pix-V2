import torch
import torch.cuda.amp as amp
from models.pix2pix_model import Pix2PixModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        self.pix2pix_model.netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pix2pix_model.netG)
        self.pix2pix_model.load_networks()
        if not opt.isTrain:
            self.pix2pix_model.eval()
        self.pix2pix_model = self.pix2pix_model.cuda()

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr
            self.scaler = amp.GradScaler()

        self.local_rank = torch.distributed.get_rank()
        self.pix2pix_model = torch.nn.parallel.DistributedDataParallel(self.pix2pix_model, 
            device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.pix2pix_model_on_one_gpu = self.pix2pix_model.module

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        with amp.autocast(self.opt.opt_level=='O1'):
            g_losses, generated = self.pix2pix_model(data, mode='generator')
            g_loss = sum(g_losses.values()).mean()
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        with amp.autocast(self.opt.opt_level=='O1'):
            d_losses = self.pix2pix_model(data, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()
        self.d_losses = d_losses

    def inference(self, data):
        images = self.pix2pix_model(data, mode='inference')
        return images

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            if self.local_rank == 0:
                print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
