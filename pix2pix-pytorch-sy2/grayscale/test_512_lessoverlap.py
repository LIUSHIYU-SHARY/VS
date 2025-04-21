from __future__ import print_function
import argparse
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from unet import define_G
from utils import is_image_file, load_img, save_img

# 解析参数
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

# 加载模型
model_path = "/home/yuqi/virtual_staining/output/pix2pix/gray/unet_40x_bright_mito_without_perceptual_checkpoint/netG_model_epoch_200.pth"
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
    transforms.Normalize((0.5, ), (0.5, ))
])

result_dir = "results/unet_plus_original"
os.makedirs(result_dir, exist_ok=True) 

# 滑动窗口参数
patch_size = 512
max_overlap = 20  # 最大允许重叠像素

def calculate_step(image_size, patch_size, max_overlap):
    required_step = patch_size - max_overlap
    total_needed = image_size - patch_size
    if total_needed <= 0:
        return 0  # 图像尺寸小于块大小
    k = total_needed // required_step
    while True:
        if k == 0:
            k = 1
        if total_needed % k == 0:
            step = total_needed // k
            if step >= required_step:
                return step
        k += 1
        if k > total_needed:
            return total_needed

def process_image(img):
    """处理2000x2000图像的核心逻辑"""
    img_tensor = transform(img).unsqueeze(0).to(device)
    output_img = torch.zeros((1, 1, 2000, 2000), device=device)
    weight_map = torch.zeros((1, 1, 2000, 2000), device=device)
    
    image_size = 2000
    step_x = calculate_step(image_size, patch_size, max_overlap)
    step_y = calculate_step(image_size, patch_size, max_overlap)
    
    positions = []
    for y in range(0, image_size - patch_size + 1, step_y):
        for x in range(0, image_size - patch_size + 1, step_x):
            positions.append((x, y))
    
    for (left, top) in positions:
        patch = img_tensor[:, :, top:top+patch_size, left:left+patch_size]
        with torch.no_grad():
            output_patch = net_g(patch)
        output_img[:, :, top:top+patch_size, left:left+patch_size] += output_patch
        weight_map[:, :, top:top+patch_size, left:left+patch_size] += 1
    
    output_img /= weight_map
    return output_img.squeeze(0).cpu()

# 处理每张图片
for image_name in image_filenames:
    print(f"Processing {image_name}...")
    img = Image.open(os.path.join(image_dir, image_name)).convert('L')
    output = process_image(img)
    save_img(output, os.path.join(result_dir, image_name))

print("测试完成，结果保存在:", result_dir)