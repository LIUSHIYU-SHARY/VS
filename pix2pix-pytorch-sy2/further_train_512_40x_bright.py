#mkdir -p result/20_bright_512_generate_images
#mkdir -p /home/yuqi/virtual_staining/output/pix2pix/mito-40x-bright/512_checkpoint
#python further_train_512_40x_bright.py --dataset ../../tmp_data/Nucleus/40x/40x-bright-512 --cuda
#cuda:0
#sy1

#2月23号训练40x bright 512模型

from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torchvision.models import vgg16  # 引入VGG模型用于感知损失
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from my_network import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data_512 import get_training_set, get_test_set

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


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='../data/Minigut')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
#change
# parser.add_argument('--save_dir', type=str, default='result/20_bright_512_generate_images', help='save generate image')
# parser.add_argument('--save_dir', type=str, default='../../output/pix2pix/Nucleus/result/2_19_40x_bright_512_generate_images', help='save generate image')
parser.add_argument('--save_dir', type=str, default='../../output/pix2pix/Nucleus/0228_512_40x_bright_generate_images', help='save generate image')
opt = parser.parse_args()

os.makedirs(opt.save_dir, exist_ok=True)

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(opt.dataset, opt.direction)
test_set = get_test_set(opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

#change #指定GPU
device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

# 加载预训练模型
if os.path.exists('../../output/pix2pix/minigut/mycheckpoint/netG_model_epoch_200.pth'):
    checkpoint_g = torch.load('../../output/pix2pix/minigut/mycheckpoint/netG_model_epoch_200.pth', map_location=device, weights_only=True)
    net_g.load_state_dict(checkpoint_g)
    net_g.load_state_dict(torch.load('../../output/pix2pix/minigut/mycheckpoint/netG_model_epoch_200.pth', map_location=device, weights_only=True))
    print("Loaded pre-trained generator weights.")
if os.path.exists('../../output/pix2pix/minigut/mycheckpoint/netD_model_epoch_200.pth'):
    # 先加载模型权重到内存中
    checkpoint_d = torch.load('../../output/pix2pix/minigut/mycheckpoint/netD_model_epoch_200.pth', map_location=device, weights_only=True)
    net_d.load_state_dict(checkpoint_d)
    net_d.load_state_dict(torch.load('../../output/pix2pix/minigut/mycheckpoint/netD_model_epoch_200.pth', map_location=device, weights_only=True))
    print("Loaded pre-trained discriminator weights.")

net_g.freeze_initial_layers()

for name, param in net_g.named_parameters():
    print(f"{name} requires_grad: {param.requires_grad}")

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

# setup optimizer
optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.AdamW(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

save_dir = opt.save_dir
writer = SummaryWriter(log_dir='logs/0228-pix2pix-512-40x-bright-Nucleus')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    i = 1
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

        # Second, G(A) = B (L1 loss)
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb

        # Third, perceptual loss
        loss_g_perceptual = perceptual_loss(fake_b, real_b) * 0.1
        
        # Fourth, consistency loss
        loss_g_consistency = consistency_loss(fake_b, real_b) * 0.1  # 权重可以调整

        # loss_g = loss_g_gan + loss_g_l1 + loss_g_perceptual+loss_g_consistency
        loss_g = loss_g_gan + loss_g_l1 + loss_g_perceptual

        loss_g.backward()
        optimizer_g.step()

        i += 1
        if i % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_Consistency: {:.4f}".format(
                epoch, iteration, len(training_data_loader), 
                loss_d.item(), loss_g.item(), loss_g_consistency.item()))

        writer.add_scalar('Loss/Generator', loss_g.item(), epoch * len(training_data_loader) + iteration)
        writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch * len(training_data_loader) + iteration)


    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    # save_image
    if epoch % 2 == 0:
        fake_sample = net_g(real_a)
        save_image(fake_sample, f"{save_dir}/sample_{epoch}.png", normalize=True)


#change 
    # checkpoint 
    if epoch % 5 == 0:
        if not os.path.exists('../../output/pix2pix/Nucleus/0228_512_40x_bright_checkpoint'):
            os.makedirs('../../output/pix2pix/Nucleus/0228_512_40x_bright_checkpoint')

        torch.save(net_g.state_dict(), '../../output/pix2pix/Nucleus/0228_512_40x_bright_checkpoint/netG_model_epoch_{}.pth'.format(epoch))
        torch.save(net_d.state_dict(), '../../output/pix2pix/Nucleus/0228_512_40x_bright_checkpoint/netD_model_epoch_{}.pth'.format(epoch))
        print("Checkpoint saved.")
        
writer.close()