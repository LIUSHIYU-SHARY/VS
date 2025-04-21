import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# 设置全局字体和样式参数
plt.rcParams.update({
    'font.family': 'sans-serif',  # 使用sans-serif字体
    'font.size': 7,          # 基础字号7pt
    'axes.titlesize': 8,     # 标题8pt
    'axes.labelsize': 7,     # 坐标轴标签7pt
    'xtick.labelsize': 6,    # x轴刻度6pt
    'ytick.labelsize': 6,    # y轴刻度6pt
    'legend.fontsize': 7,    # 图例7pt
    'lines.linewidth': 0.8,  # 线条宽度0.8pt
    'axes.linewidth': 0.6,   # 轴线宽度0.6pt
    'grid.linewidth': 0.4    # 网格线宽度0.4pt
})

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
save_dir = "radar_figures_2"
os.makedirs(save_dir, exist_ok=True)

# 绘图函数
def save_radar(data_subset, title, filename):
    fig = plt.figure(figsize=(10, 7), facecolor='white')  # 白色背景
    ax = fig.add_subplot(111, polar=True)
    
    # 设置白色背景和网格线
    ax.set_facecolor('white')
    ax.grid(color='gray', linestyle='-', linewidth=0.4, alpha=0.5)
    
    # 绘制数据
    for i in range(len(data_subset)):
        values = data_subset.iloc[i, 1:].tolist()
        values += values[:1]
        ax.plot(angles, values, label=data_subset['Name'].iloc[i], alpha=0.7, linewidth=0.8)
        ax.fill(angles, values, alpha=0.05)

    # 绘制平均值
    mean_values = data_subset.iloc[:, 1:].mean().tolist()
    mean_values += mean_values[:1]
    ax.plot(angles, mean_values, label='Mean', linestyle='--', linewidth=1.2, color='black')

    # 设置标签和标题
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=8, pad=15)  # 标题8pt
    
    # 调整图例位置和样式
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), 
                      frameon=True, framealpha=0.9)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(0.4)
    
    plt.tight_layout()
    
    # 保存图像（300DPI，白色背景）
    plt.savefig(os.path.join(save_dir, filename), 
                dpi=300, 
                facecolor='white',
                edgecolor='none',
                bbox_inches='tight',
                transparent=False)
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