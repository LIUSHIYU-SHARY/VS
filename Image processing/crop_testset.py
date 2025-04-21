import os
import cv2
import numpy as np
from tqdm import tqdm

def crop_images(input_folder, output_folder, crop_size=512, stride=256):
    """
    将 input_folder 中的 2000x2000 图像裁剪为 512x512 小块，并保存到 output_folder。
    :param input_folder: 输入文件夹
    :param output_folder: 输出文件夹
    :param crop_size: 裁剪的图像大小 (默认 512)
    :param stride: 步长 (默认 256)，用于控制重叠区域
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_file}, could not be loaded.")
            continue
        
        height, width = img.shape[:2]
        if height != 2000 or width != 2000:
            print(f"Skipping {img_file}, expected 2000x2000 but got {width}x{height}.")
            continue
        
        # 计算裁剪起始位置
        crop_id = 0
        for y in range(0, height - crop_size + 1, stride):
            for x in range(0, width - crop_size + 1, stride):
                cropped_img = img[y:y + crop_size, x:x + crop_size]
                save_name = f"{os.path.splitext(img_file)[0]}_crop{crop_id}.png"
                cv2.imwrite(os.path.join(output_folder, save_name), cropped_img)
                crop_id += 1
        
    print("Processing complete.")

# 示例调用
input_folder = "40x-bright-512/test/b"  # 修改为你的输入文件夹
output_folder = "my_testset/b"  # 修改为你的输出文件夹
crop_images(input_folder, output_folder, crop_size=512, stride=256)
