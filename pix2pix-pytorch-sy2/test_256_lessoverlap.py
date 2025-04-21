# # python test_256_lessoverlap.py --dataset ../../tmp_data/Nucleus/20x/20x-phase-256

from __future__ import print_function
import argparse
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from networks import define_G
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
model_path = "/home/yuqi/virtual_staining/output/pix2pix/Nucleus/0225_256_20x_phase_checkpoint/netG_model_epoch_200.pth"
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
state_dict = torch.load(model_path, map_location=device)
net_g.load_state_dict(state_dict)
net_g.to(device)
net_g.eval()

# 数据路径
image_dir = f"{opt.dataset}/test/{'a' if opt.direction == 'a2b' else 'b'}/"
print(image_dir)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

result_dir = "Nucleus_result/0225_256_20x_phase"
os.makedirs(result_dir, exist_ok=True) 

# 滑动窗口参数
patch_size = 256  # 修改为 256x256
max_overlap = 10  # 最大允许重叠像素

def calculate_step(image_size, patch_size, max_overlap):
    """
    计算滑动窗口的步长，确保重叠不超过 max_overlap。
    """
    required_step = patch_size - max_overlap
    return required_step

def process_image(img):
    """处理2000x2000图像的核心逻辑"""
    img_tensor = transform(img).unsqueeze(0).to(device)
    output_img = torch.zeros((1, 3, 2000, 2000), device=device)
    weight_map = torch.zeros((1, 3, 2000, 2000), device=device)
    
    image_size = 2000
    step = calculate_step(image_size, patch_size, max_overlap)
    
    # 生成块的位置
    positions = []
    for y in range(0, image_size - patch_size + 1, step):
        for x in range(0, image_size - patch_size + 1, step):
            positions.append((x, y))
    
    # 确保最右侧和最底部的部分也被处理
    if (image_size - patch_size) % step != 0:
        for y in range(0, image_size - patch_size + 1, step):
            positions.append((image_size - patch_size, y))
        for x in range(0, image_size - patch_size + 1, step):
            positions.append((x, image_size - patch_size))
        positions.append((image_size - patch_size, image_size - patch_size))
    
    # 处理每个块
    for (left, top) in positions:
        patch = img_tensor[:, :, top:top+patch_size, left:left+patch_size]
        with torch.no_grad():
            output_patch = net_g(patch)
        output_img[:, :, top:top+patch_size, left:left+patch_size] += output_patch
        weight_map[:, :, top:top+patch_size, left:left+patch_size] += 1
    
    # 归一化输出图像
    output_img /= weight_map
    return output_img.squeeze(0).cpu()

# 处理每张图片
for image_name in image_filenames:
    print(f"Processing {image_name}...")
    img = Image.open(os.path.join(image_dir, image_name)).convert('RGB')
    output = process_image(img)
    save_img(output, os.path.join(result_dir, image_name))

print("测试完成，结果保存在:", result_dir)