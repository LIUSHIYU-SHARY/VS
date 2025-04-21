import os
import cv2
import numpy as np
import pandas as pd
import _utils_calculate_psnr_ssim as util
from pytorch_fid import fid_score
import matplotlib.pyplot as plt

def average_metrics(folder1, folder2, output_pdf="0320_512_20x_phase.pdf"):
    filenames1 = {os.path.basename(f) for f in os.listdir(folder1) if f.endswith('.tif')}
    filenames2 = {os.path.basename(f) for f in os.listdir(folder2) if f.endswith('.tif')}
    
    common_filenames = filenames1.intersection(filenames2)
    
    results = []
    total_psnr, total_ssim, total_lpips, total_dice, total_pearson = 0, 0, 0, 0, 0
    count = 0
    
    for filename in common_filenames:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is not None and img2 is not None:
            crop_border = 0
            psnr = util.calculate_psnr(img1, img2, crop_border, test_y_channel=True)
            ssim = util.calculate_ssim(img1, img2, crop_border, test_y_channel=True)
            lpips = util.calculate_lpips(img1, img2)
            dice = util.dice_coefficient(img1, img2)
            pearson = util.pearson_correlation(img1, img2)
            
            results.append([filename, psnr, ssim, lpips, dice, pearson])
            
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            total_dice += dice
            total_pearson += pearson
            count += 1
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        avg_dice = total_dice / count
        avg_pearson = total_pearson / count

        results.append(["Average", avg_psnr, avg_ssim, avg_lpips, avg_dice, avg_pearson])
        
        # 创建 DataFrame
        df = pd.DataFrame(results, columns=["Filename", "PSNR", "SSIM", "LPIPS", "Dice", "Pearson"])
        
        # 保存为 PDF
        save_table_as_pdf(df, output_pdf)

        print(f"Metrics saved to {output_pdf}")
        
    else:
        print("No matching files found in both folders.")


def save_table_as_pdf(df, output_pdf):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.4 + 2))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

    # 调整表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3, 4, 5])

    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()


# Example usage

folder1 = '/home/yuqi/virtual_staining/project_code/pix2pix-pytorch-sy/Nucleus_result/0320_512_20x_phase'
folder2 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase-512/test/b'

fid_value = util.calculate_fid(folder2, folder1)
average_metrics(folder1, folder2, output_pdf="0320_512_20x_phase.pdf")

print(f'FID value: {fid_value:.6f}')

