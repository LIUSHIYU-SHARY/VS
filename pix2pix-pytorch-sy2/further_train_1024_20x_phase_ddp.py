from __future__ import print_function
import argparse
import os
from math import log10
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.models import vgg16
from torch.utils.tensorboard import SummaryWriter

from my_network_test import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data_1024 import get_training_set, get_test_set


def consistency_loss(fake_image, real_image):
    def ssim_loss(x, y, kernel_size=11, sigma=1.5):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        channels = x.shape[1]
        kernel = torch.ones((channels, 1, kernel_size, kernel_size)).to(x.device) / (kernel_size * kernel_size)
        mu_x = F.conv2d(x, kernel, padding=kernel_size // 2, groups=channels)
        mu_y = F.conv2d(y, kernel, padding=kernel_size // 2, groups=channels)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.conv2d(x * x, kernel, padding=kernel_size // 2, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, kernel, padding=kernel_size // 2, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=kernel_size // 2, groups=channels) - mu_xy
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return 1 - ssim.mean()

    def feature_consistency_loss(x, y):
        losses = []
        for scale in [1, 0.5, 0.25]:
            curr_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            curr_y = F.interpolate(y, scale_factor=scale, mode='bilinear', align_corners=False)
            grad_x_x = curr_x[:, :, :, 1:] - curr_x[:, :, :, :-1]
            grad_x_y = curr_x[:, :, 1:, :] - curr_x[:, :, :-1, :]
            grad_y_x = curr_y[:, :, :, 1:] - curr_y[:, :, :, :-1]
            grad_y_y = curr_y[:, :, 1:, :] - curr_y[:, :, :-1, :]
            loss_grad_x = F.l1_loss(grad_x_x, grad_y_x) + F.l1_loss(grad_x_y, grad_y_y)
            losses.append(loss_grad_x)
        return sum(losses)

    ssim = ssim_loss(fake_image, real_image)
    feat_cons = feature_consistency_loss(fake_image, real_image)
    return ssim + 0.1 * feat_cons

def perceptual_loss(fake, real, vgg):
    fake_features = vgg(fake)
    real_features = vgg(real)
    return nn.MSELoss()(fake_features, real_features)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12375'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size, opt):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set(opt.dataset, opt.direction)
    test_set = get_test_set(opt.dataset, opt.direction)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    training_data_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=opt.batch_size, num_workers=opt.threads, pin_memory=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.threads)

    device = torch.device(f"cuda:{rank}")

    net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02).to(device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic').to(device)

    if os.path.exists('../../output/pix2pix/Nucleus/0319_minigut_checkpoint/netG_model_epoch_200.pth'):
        checkpoint_g = torch.load('../../output/pix2pix/Nucleus/0319_minigut_checkpoint/netG_model_epoch_200.pth', map_location='cpu')
        net_g.load_state_dict(checkpoint_g)
        print("Loaded pre-trained generator weights.")

    if os.path.exists('../../output/pix2pix/Nucleus/0319_minigut_checkpoint/netD_model_epoch_200.pth'):
        checkpoint_d = torch.load('../../output/pix2pix/Nucleus/0319_minigut_checkpoint/netD_model_epoch_200.pth', map_location='cpu')
        net_d.load_state_dict(checkpoint_d)
        print("Loaded pre-trained discriminator weights.")

    net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_g)
    net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_d)
    net_g = nn.parallel.DistributedDataParallel(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = nn.parallel.DistributedDataParallel(net_d, device_ids=[rank], find_unused_parameters=True)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    vgg = vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.AdamW(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    save_dir = opt.save_dir
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir='logs/0331-pix2pix-1024-20x-phase-Nucleus')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        net_g.train()
        net_d.train()
        train_sampler.set_epoch(epoch)
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            fake_b = net_g(real_a)

            optimizer_d.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d(real_ab)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)
            loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
            loss_g_perceptual = perceptual_loss(fake_b, real_b, vgg) * 0.1
            loss_g_consistency = consistency_loss(fake_b, real_b) * 0.1
            loss_g = loss_g_gan + loss_g_l1 + loss_g_perceptual + loss_g_consistency
            loss_g.backward()
            optimizer_g.step()

            if rank == 0 and iteration % 10 == 0:
                print(f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} Loss_Consistency: {loss_g_consistency.item():.4f}")
                writer.add_scalar('Loss/Generator', loss_g.item(), epoch * len(training_data_loader) + iteration)
                writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch * len(training_data_loader) + iteration)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        if rank == 0:
            net_g.eval()
            avg_psnr = 0
            with torch.no_grad():
                for batch in testing_data_loader:
                    input, target = batch[0].to(device), batch[1].to(device)
                    prediction = net_g(input)
                    mse = criterionMSE(prediction, target)
                    psnr = 10 * log10(1 / mse.item())
                    avg_psnr += psnr
            print(f"===> Avg. PSNR: {avg_psnr / len(testing_data_loader):.4f} dB")
            if epoch % 2 == 0:
                fake_sample = net_g(real_a)
                save_image(fake_sample, f"{save_dir}/sample_{epoch}.png", normalize=True)
            if epoch % 5 == 0:
                ckpt_path = '../../output/pix2pix/Nucleus/0331_1024_20x_phase_checkpoint'
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(net_g.module.state_dict(), f'{ckpt_path}/netG_model_epoch_{epoch}.pth')
                torch.save(net_d.module.state_dict(), f'{ckpt_path}/netD_model_epoch_{epoch}.pth')
                print("Checkpoint saved.")

    if rank == 0:
        writer.close()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-ddp')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--direction', type=str, default='a2b')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epoch_count', type=int, default=1)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--niter_decay', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_policy', type=str, default='lambda')
    parser.add_argument('--lr_decay_iters', type=int, default=50)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lamb', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='../../output/pix2pix/Nucleus/0331_1024_20x_phase_generate_images')
    opt = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, opt,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()