import os
import cv2
import numpy as np
import _utils_calculate_psnr_ssim as util
from pytorch_fid import fid_score

def calculate_mse(img1, img2):
    """计算均方误差（MSE）"""
    if img1.shape != img2.shape:
        raise ValueError("两张图像的尺寸不匹配！")
    
    mse = np.mean((img1.astype("float32") - img2.astype("float32")) ** 2)
    return mse

def average_metrics(folder1, folder2):
    filenames1 = {os.path.basename(f) for f in os.listdir(folder1) if f.endswith('.jpg')}
    filenames2 = {os.path.basename(f) for f in os.listdir(folder2) if f.endswith('.jpg')}
    
    common_filenames = filenames1.intersection(filenames2)
    total_psnr, total_ssim, total_lpips, total_dice, total_pearson ,total_mse = 0, 0, 0, 0, 0, 0
    count = 0
    
    for filename in common_filenames:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)
        
        img1 = cv2.imread(img1_path)
                          # transform into 3 channel image
        img2 = cv2.imread(img2_path)
        
        if img1 is not None and img2 is not None:
            
            crop_border = 0
            psnr = util.calculate_psnr(img1, img2, crop_border, test_y_channel=True)
            ssim = util.calculate_ssim(img1, img2, crop_border, test_y_channel=True)
            lpips = util.calculate_lpips(img1, img2)
            dice = util.dice_coefficient(img1, img2)
            pearson = util.pearson_correlation(img1, img2)
            mse = calculate_mse(img1, img2)  # 计算 MSE
            
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            total_dice += dice
            total_pearson += pearson
            total_mse += mse
            count += 1
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        avg_dice = total_dice / count
        avg_pearson = total_pearson / count
        avg_mse = total_mse / count
        
        print(f'Average PSNR: {avg_psnr:.6f} dB')
        print(f'Average SSIM: {avg_ssim:.6f}')
        print(f'Average LPIPS: {avg_lpips:.6f}')
        print(f'Average Dice: {avg_dice:.6f}')
        print(f'Average Pearson Correlation: {avg_pearson:.6f}')
        print(f'Average MSE: {avg_mse:.6f}')  # 输出 MSE 结果
    else:
        print("No matching files found in both folders.")


# Example usage
# folder1 = '../autodl-tmp/easy/generate_image/Ugatit/test'
# folder2 = '../autodl-tmp/easy/generate_image/Ugatit/compare'

folder1 = '/home/yuqi/virtual_staining/project_code/pix2pix-pytorch-sy/Nucleus_result/0319_Minigut'
folder2 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase-512/test/b'

# folder1 = '/home/yuqi/virtual_staining/project_code/grayscale/results/unet_reorganize'
# folder2 = '/home/yuqi/virtual_staining/tmp_data/Mito/40x-bright-512/test/b'

fid_value=util.calculate_fid(folder2, folder1)
average_metrics(folder1, folder2)
print(f'fid value: {fid_value:.6f}')