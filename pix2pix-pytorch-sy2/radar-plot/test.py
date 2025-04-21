
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# 数据定义
data = {
    "Name": [
        "256_20x_bright", "256_20x_phase", "512_20x_bright",
        "512_20x_phase", "256_40x_bright", "512_40x_bright"
    ],
    "PSNR (dB)": [23.021265, 21.838257, 23.954990, 25.532681, 19.024965, 21.717903],
    "SSIM": [0.586082, 0.486519, 0.572697, 0.680411, 0.482385, 0.553268],
    "LPIPS": [0.175383, 0.167502, 0.151413, 0.110825, 0.300593, 0.207667],
    "DICE": [0.980361, 0.998487, 0.991685, 0.998499, 0.958011, 0.952446],
    "Pearson": [0.299133, 0.394604, 0.391666, 0.555677, 0.270225, 0.469103],
    "FID": [109.382962, 120.740987, 125.331269, 68.689233, 412.108408, 238.599497]
}

df = pd.DataFrame(data)

# 归一化处理
normalized_df = df.copy()
normalized_df['LPIPS'] = 1 - normalized_df['LPIPS'] / normalized_df['LPIPS'].max()
normalized_df['FID'] = 1 - normalized_df['FID'] / normalized_df['FID'].max()

for col in ['PSNR (dB)', 'SSIM', 'DICE', 'Pearson']:
    normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# 雷达图设置
categories = ['PSNR (dB)', 'SSIM', 'LPIPS', 'DICE', 'Pearson', 'FID']
angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
angles += angles[:1]

# 创建保存目录
save_dir = "radar_figures"
os.makedirs(save_dir, exist_ok=True)

# 绘图函数
def save_radar(data_subset, title, filename):
    plt.figure(figsize=(10, 7))
    for i in range(len(data_subset)):
        values = data_subset.iloc[i, 1:].tolist()
        values += values[:1]
        plt.polar(angles, values, label=data_subset['Name'].iloc[i], alpha=0.7)

    mean_values = data_subset.iloc[:, 1:].mean().tolist()
    mean_values += mean_values[:1]
    plt.polar(angles, mean_values, label='Mean', linestyle='--', linewidth=2, color='black')

    plt.xticks(angles[:-1], categories)
    plt.title(title, size=13)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, transparent=True)
    plt.close()

# 图数据子集
subsets = {
    "radar_all.png": normalized_df,
    "radar_bright.png": normalized_df[normalized_df['Name'].str.contains('bright')].reset_index(drop=True),
    "radar_phase.png": normalized_df[normalized_df['Name'].str.contains('phase')].reset_index(drop=True),
    "radar_resolution_256.png": normalized_df[normalized_df['Name'].str.contains('256')].reset_index(drop=True),
    "radar_resolution_512.png": normalized_df[normalized_df['Name'].str.contains('512')].reset_index(drop=True),
    "radar_magnification_20x.png": normalized_df[normalized_df['Name'].str.contains('20x')].reset_index(drop=True),
    "radar_magnification_40x.png": normalized_df[normalized_df['Name'].str.contains('40x')].reset_index(drop=True)
}

# 生成所有雷达图
for filename, subset in subsets.items():
    title = filename.replace("radar_", "").replace(".png", "").replace("_", " ").title()
    save_radar(subset, f"Performance Comparison - {title}", filename)
