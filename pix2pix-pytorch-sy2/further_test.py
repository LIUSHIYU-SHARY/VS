from __future__ import print_function
import argparse
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from networks import define_G  # Adjust this import according to your actual model definition file
from utils import is_image_file, load_img, save_img

# 解析参数
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

# 加载模型
# model_path = "./result/my_checkpoints/netG_model_epoch_100.pth"
model_path = "/home/yuqi/virtual_staining/output/pix2pix/minigut/mycheckpoint/netG_model_epoch_174.pth"
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
state_dict = torch.load(model_path, map_location=device)
net_g.load_state_dict(state_dict)
net_g.to(device)
net_g.eval()

# 数据路径
if opt.direction == "a2b":
    image_dir = "{}/test/a/".format(opt.dataset)
else:
    image_dir = "{}/test/b/".format(opt.dataset)
    
print(image_dir)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

result_dir = "freeze_pretrained_174"
os.makedirs(result_dir, exist_ok=True)

# 滑动窗口参数
patch_size = 256  # 裁剪块大小
stride = 64  # 滑动窗口步长 (25% 重叠)

# 处理每张图片
for image_name in image_filenames:
    img = load_img(os.path.join(image_dir, image_name))  # 加载 PIL Image
    img = img.resize((512, 512), Image.BICUBIC)  # 确保尺寸一致

    img_tensor = transform(img)  # 转换为张量
    img_tensor = img_tensor.unsqueeze(0).to(device)  # 增加 batch 维度

    # 初始化输出图像及权重矩阵
    output_img = torch.zeros_like(img_tensor)
    weight_map = torch.zeros_like(img_tensor)

    # 遍历子块
    for i in range(0, 512 - patch_size + 1, stride):
        for j in range(0, 512 - patch_size + 1, stride):
            # 提取子块
            patch = img_tensor[:, :, i:i+patch_size, j:j+patch_size]
            with torch.no_grad():
                output_patch = net_g(patch)  # 通过模型生成输出
            
            # 累加结果
            output_img[:, :, i:i+patch_size, j:j+patch_size] += output_patch
            weight_map[:, :, i:i+patch_size, j:j+patch_size] += 1  # 记录叠加次数

    # 计算平均值
    output_img /= weight_map
    output_img = output_img.squeeze(0).cpu()  # 移除 batch 维度

    # 保存结果
    save_img(output_img, os.path.join(result_dir, image_name))

print("测试完成，结果保存在:", result_dir)
