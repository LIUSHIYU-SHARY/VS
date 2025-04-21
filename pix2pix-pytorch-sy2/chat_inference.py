import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from my_network import define_G

# 设置参数
model_path = "output/netG_model_epoch_185.pth"
input_folder = "/home/yuqi/virtual_staining/tmp_data/Mito/40x-bright-512/test/a"  # 需要处理的文件夹
output_folder = "result/test_40x_512to256_result"  # 输出文件夹
patch_size = 256
stride = 128  # 步长，减少拼接痕迹

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预处理变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载模型
net_g = define_G(input_nc=3, output_nc=3, ngf=64, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id=device)
net_g.load_state_dict(torch.load(model_path, map_location=device))
net_g.to(device)
net_g.eval()

# 处理整个文件夹
os.makedirs(output_folder, exist_ok=True)
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.tif'))]

for image_file in image_files:
    input_image_path = os.path.join(input_folder, image_file)
    output_image_path = os.path.join(output_folder, image_file)

    # 读取测试图像
    image = Image.open(input_image_path).convert("RGB")
    w, h = image.size
    assert w == 2000 and h == 2000, f"{image_file} 的尺寸必须是 2000x2000"

    # 计算 patch 数量
    num_patches_w = (w - patch_size) // stride + 1
    num_patches_h = (h - patch_size) // stride + 1
    
    # 初始化输出图像和权重矩阵
    output_image = np.zeros((h, w, 3), dtype=np.float32)
    weight_map = np.zeros((h, w, 3), dtype=np.float32)
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            left = j * stride
            upper = i * stride
            right = left + patch_size
            lower = upper + patch_size

            # 裁剪 256x256 的 patch
            patch = image.crop((left, upper, right, lower))
            patch_tensor = transform(patch).unsqueeze(0).to(device)

            # 通过模型生成
            with torch.no_grad():
                output_patch = net_g(patch_tensor)
                output_patch = output_patch.squeeze(0).cpu().detach()

            # 反归一化
            output_patch = output_patch * 0.5 + 0.5
            output_patch = output_patch.permute(1, 2, 0).numpy()
            output_patch = (output_patch * 255).astype(np.float32)
            
            # 叠加到输出图像
            output_image[upper:lower, left:right, :] += output_patch
            weight_map[upper:lower, left:right, :] += 1
    
    # 归一化，防止重叠区域过亮
    output_image /= np.maximum(weight_map, 1)
    output_image = output_image.astype(np.uint8)

    # 保存最终的 2000x2000 生成图像
    output_pil_image = Image.fromarray(output_image)
    output_pil_image.save(output_image_path)
    print(f"{image_file} 处理完成，已保存到 {output_image_path}")
