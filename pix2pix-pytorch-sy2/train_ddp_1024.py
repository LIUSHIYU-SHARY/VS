# python train_ddp_1024.py --dataset ../../tmp_data/Nuclear-classification/20x/20x-phase-1024 --cuda
from __future__ import print_function
import argparse
import os
from math import log10
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from torchvision.models import vgg16
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd
torch.autograd.set_detect_anomaly(True)  # 添加异常检测

from my_network_test import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data_1024 import get_training_set, get_test_set


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12375'
    # os.environ['MASTER_PORT'] = '34999'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def consistency_loss(fake_image, real_image):
    # 结构相似性损失
    def ssim_loss(x, y, kernel_size=11, sigma=1.5):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 动态匹配输入通道数
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
    
    # 多尺度特征一致性损失
    def feature_consistency_loss(x, y):
        losses = []
        for scale in [1, 0.5, 0.25]:
            if scale != 1:
                curr_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                curr_y = F.interpolate(y, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                curr_x = x
                curr_y = y
            
            # 计算梯度
            grad_x_x = curr_x[:, :, :, 1:] - curr_x[:, :, :, :-1]
            grad_x_y = curr_x[:, :, 1:, :] - curr_x[:, :, :-1, :]
            grad_y_x = curr_y[:, :, :, 1:] - curr_y[:, :, :, :-1]
            grad_y_y = curr_y[:, :, 1:, :] - curr_y[:, :, :-1, :]
            
            # 计算梯度差异
            loss_grad_x = F.l1_loss(grad_x_x, grad_y_x) + F.l1_loss(grad_x_y, grad_y_y)
            losses.append(loss_grad_x)
        
        return sum(losses)
    
    # 组合损失
    ssim = ssim_loss(fake_image, real_image)
    feat_cons = feature_consistency_loss(fake_image, real_image)
    
    return ssim + 0.1 * feat_cons

def train(rank, world_size, opt):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set(opt.dataset, opt.direction)
    test_set = get_test_set(opt.dataset, opt.direction)
    
    train_sampler = DistributedSampler(train_set, 
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=True)
    
    training_data_loader = DataLoader(dataset=train_set, 
                                    num_workers=opt.threads,
                                    batch_size=opt.batch_size,
                                    sampler=train_sampler,
                                    pin_memory=True)
    
    testing_data_loader = DataLoader(dataset=test_set,
                                   num_workers=opt.threads,
                                   batch_size=opt.test_batch_size,
                                   shuffle=False)

    device = torch.device(rank)

    # 创建模型并移动到GPU
    net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

    # 确保模型参数在正确的设备上
    net_g = net_g.to(device)
    net_d = net_d.to(device)
    
    # # === 加载预训练模型（在 DDP 包裹之前）===
    # if rank == 0:
    #     print("===> Loading pre-trained models if available...")

    # g_ckpt_path = '../../output/pix2pix/Nucleus/0319_minigut_checkpoint/netG_model_epoch_200.pth'
    # d_ckpt_path = '../../output/pix2pix/Nucleus/0319_minigut_checkpoint/netD_model_epoch_200.pth'

    # if os.path.exists(g_ckpt_path):
    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # 多GPU下映射
    #     checkpoint_g = torch.load(g_ckpt_path, map_location=map_location)
    #     net_g.load_state_dict(checkpoint_g)
    #     if rank == 0:
    #         print("Loaded pre-trained generator weights.")

    # if os.path.exists(d_ckpt_path):
    #     checkpoint_d = torch.load(d_ckpt_path, map_location=map_location)
    #     net_d.load_state_dict(checkpoint_d)
    #     if rank == 0:
    #         print("Loaded pre-trained discriminator weights.")

    # 同步BatchNorm
    net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_g)
    net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_d)

    # 转换为DDP模型
    net_g = nn.parallel.DistributedDataParallel(net_g, 
                                               device_ids=[rank],
                                               find_unused_parameters=True,
                                               broadcast_buffers=False)
    net_d = nn.parallel.DistributedDataParallel(net_d, 
                                               device_ids=[rank],
                                               find_unused_parameters=True,
                                               broadcast_buffers=False)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    
    
    # 感知损失
    vgg = vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False  # 冻结VGG权重
    
    def perceptual_loss(fake, real):
        fake_features = vgg(fake)
        real_features = vgg(real)
        return nn.MSELoss()(fake_features, real_features)


    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    save_dir = '../../output/pix2pix/Nucleus/0409_1024_20x_phase_generate_images'
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        net_g.train()
        net_d.train()
        train_sampler.set_epoch(epoch)
        
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b = batch[0].to(device), batch[1].to(device)

            ######################
            # Update D network
            ######################
            optimizer_d.zero_grad(set_to_none=True)

            # 生成假图片，不需要梯度
            with torch.no_grad():
                fake_b = net_g(real_a)
            
            # 训练判别器
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d(real_ab)
            loss_d_real = criterionGAN(pred_real, True)

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            ######################
            # Update G network
            ######################
            optimizer_g.zero_grad(set_to_none=True)

            # 重新生成假图片
            fake_b = net_g(real_a)
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)
            loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
            
            loss_g_perceptual = perceptual_loss(fake_b, real_b) * 0.1
            
            loss_g_consistency = consistency_loss(fake_b, real_b) * 0.1  # 权重可以调整
            
            #loss_g = loss_g_gan + loss_g_l1
            loss_g = loss_g_gan + loss_g_l1 + loss_g_perceptual+loss_g_consistency
            
            loss_g.backward()
            optimizer_g.step()

            if rank == 0 and iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # 测试阶段
        if rank == 0:
            net_g.eval()
            avg_psnr = 0
            torch.cuda.empty_cache()
            with torch.no_grad():
                for batch in testing_data_loader:
                    input, target = batch[0].to(device), batch[1].to(device)
                    prediction = net_g(input)
                    mse = criterionMSE(prediction, target)
                    psnr = 10 * log10(1 / mse.item())
                    avg_psnr += psnr
                print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

                # 保存示例图片
                # if epoch % 2 == 0:
                #     fake_sample = net_g(real_a)
                #     save_image(fake_sample, f"{save_dir}/sample_{epoch}.png", normalize=True)
                if epoch % 1 == 0:
                    fake_sample = net_g(real_a)
                    fake_sample_cpu = fake_sample.cpu()
                    save_image(fake_sample_cpu, f"{save_dir}/sample_{epoch}.png", normalize=True)
                    del fake_sample, fake_sample_cpu
                    torch.cuda.empty_cache()


            # 保存检查点
            if epoch % 1 == 0:
                if not os.path.exists('../../output/pix2pix/Nucleus/0409_1024_20x_phase_checkpoints'):
                    os.makedirs('../../output/pix2pix/Nucleus/0409_1024_20x_phase_checkpoints')
                
                torch.save(net_g.module.state_dict(), 
                         f'../../output/pix2pix/Nucleus/0409_1024_20x_phase_checkpoints/netG_model_epoch_{epoch}.pth')
                torch.save(net_d.module.state_dict(), 
                         f'../../output/pix2pix/Nucleus/0409_1024_20x_phase_checkpoints/netD_model_epoch_{epoch}.pth')
                print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', required=True, help='../data/in_silico')
    parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--save_dir', type=str, default='output/generate_image', help='save generate image')
    
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    #world_size = torch.cuda.device_count()
    world_size = 3
    mp.spawn(train,
             args=(world_size, opt,),
             nprocs=world_size,
             join=True)
