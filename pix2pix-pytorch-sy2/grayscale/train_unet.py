from __future__ import print_function
import argparse
import os
from math import log10
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torchvision.models import vgg16  
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import structural_loss,perceptual_loss

from unet import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='../data/in_silico')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
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
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--save_dir', type=str, default='../../output/pix2pix/in-silico/generate_image', help='save generate image')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
#train_set = get_training_set(root_path + opt.dataset, opt.direction)
#test_set = get_test_set(root_path + opt.dataset, opt.direction)
train_set = get_training_set(opt.dataset, opt.direction)
test_set = get_test_set(opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:3" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)


# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

save_dir='../../output/pix2pix/gray/unet_40x_bright_mito_without_perceptual_image'
#writer = SummaryWriter(log_dir='logs/pix2pix-512-40x-bright-mito')

#num_batch = 6  # 总共运行 100 个 batch
#start_time = time.time()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
#for epoch in range(num_batch):
    # train
    i=1
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
        # Third, perceptual loss
        # loss_g_perceptual = perceptual_loss(fake_b, real_b,device) * 0.1
        
        # Fourth, consistency loss
        loss_g_structural = structural_loss(fake_b, real_b) * 0.1  # 权重可以调整
        loss_g = loss_g_gan + loss_g_l1 + loss_g_structural
        #loss_g = loss_g_gan + loss_g_l1 + loss_g_perceptual+loss_g_structural
        #loss_g = loss_g_gan + loss_g_l1
 
        loss_g.backward()

        optimizer_g.step()
        i=i+1
        if i%10==0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
                
#
#        writer.add_scalar('Loss/Generator', loss_g.item(), epoch * len(training_data_loader) + iteration)
#        writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch * len(training_data_loader) + iteration)
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    torch.cuda.empty_cache()
    #save_image
    if epoch % 2 ==0:
        with torch.no_grad():  # 避免计算梯度
            fake_sample = net_g(real_a)
            # 将tensor转移到CPU并释放GPU内存
            fake_sample_cpu = fake_sample.cpu()
            save_image(fake_sample_cpu, f"{save_dir}/sample_{epoch}.png", normalize=True)
            del fake_sample, fake_sample_cpu
            torch.cuda.empty_cache()
        
    #checkpoint
    if epoch % 5 == 0:
        if not os.path.exists('../../output/pix2pix/gray/unet_40x_bright_mito_without_perceptual_checkpoint'):
            os.mkdir('../../output/pix2pix/gray/unet_40x_bright_mito_without_perceptual_checkpoint')

        torch.save(net_g.state_dict(), '../../output/pix2pix/gray/unet_40x_bright_mito_without_perceptual_checkpoint/netG_model_epoch_{}.pth'.format(epoch))
        torch.save(net_d.state_dict(), '../../output/pix2pix/gray/unet_40x_bright_mito_without_perceptual_checkpoint/netD_model_epoch_{}.pth'.format(epoch))
        

