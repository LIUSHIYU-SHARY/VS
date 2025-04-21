import numpy as np
from PIL import Image
from torchvision.models import vgg16 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    #img = img.resize((256, 256), Image.BICUBIC)
    return img

#
#def save_img(image_tensor, filename):
#    image_numpy = image_tensor.float().numpy()
#    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#    image_numpy = image_numpy.clip(0, 255)
#    image_numpy = image_numpy.astype(np.uint8)
#    image_pil = Image.fromarray(image_numpy)
#    image_pil.save(filename)
#    print("Image saved as {}".format(filename))

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    
    # 如果是单通道图像，去掉多余的维度
    if image_numpy.shape[0] == 1:
        image_numpy = image_numpy.squeeze()  # 移除所有大小为1的维度
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # 应用对比度增强
    image_numpy = (image_numpy + 1) / 2.0  # 转换到[0,1]范围
    image_numpy = np.power(image_numpy, 0.9)  # 微调对比度
    
    image_numpy = image_numpy * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    
    # 保存为更高质量的图像格式
    image_pil = Image.fromarray(image_numpy, mode='L')  # 指定mode='L'表示灰度图
    image_pil.save(filename, quality=95)  # 增加保存质量


def structural_loss(fake_image, real_image):
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



def perceptual_loss(fake, real, device):
    vgg = vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False  # 冻结VGG权重
    fake_3ch = fake.repeat(1,3,1,1)                               # turn to three channel
    real_3ch = real.repeat(1,3,1,1)
    fake_features = vgg(fake_3ch)
    real_features = vgg(real_3ch)
    return nn.MSELoss()(fake_features, real_features)