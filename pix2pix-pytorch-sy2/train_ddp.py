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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd
torch.autograd.set_detect_anomaly(True)  # 添加异常检测

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12375'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    save_dir = 'output/generate_image'
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
            loss_g = loss_g_gan + loss_g_l1
            
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
                if epoch % 1 == 0:
                    fake_sample = net_g(real_a)
                    fake_sample_cpu = fake_sample.cpu()
                    save_image(fake_sample_cpu, f"{save_dir}/sample_{epoch}.png", normalize=True)
                    del fake_sample, fake_sample_cpu
                    torch.cuda.empty_cache()

            # 保存检查点
            if epoch % 1 == 0:
                if not os.path.exists('../../output/pix2pix/in-silico/mycheckpoint'):
                    os.makedirs('../../output/pix2pix/in-silico/mycheckpoint')
                
                torch.save(net_g.module.state_dict(), 
                         f'../../output/pix2pix/in-silico/mycheckpoint/netG_model_epoch_{epoch}.pth')
                torch.save(net_d.module.state_dict(), 
                         f'../../output/pix2pix/in-silico/mycheckpoint/netD_model_epoch_{epoch}.pth')
                print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', required=True, help='../data/in_silico')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
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

    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, opt,),
             nprocs=world_size,
             join=True)