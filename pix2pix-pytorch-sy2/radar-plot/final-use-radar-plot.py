import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap

# 设置全局字体和样式参数
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4
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
save_dir = "radar_figures_final(no mean)"
os.makedirs(save_dir, exist_ok=True)

# 定义统一的颜色映射
unique_names = df['Name'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_names)))
name_to_color = dict(zip(unique_names, colors))

# 绘图函数
def save_radar(data_subset, title, filename):
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    
    # 设置白色背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 设置网格线
    ax.grid(color='lightgray', linestyle='-', linewidth=0.4, alpha=0.7)
    
    # 添加径向标尺
    ax.set_rgrids(
        radii=[0.2, 0.4, 0.6, 0.8, 1.0],
        angle=45,
        labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
        fontsize=6,
        color='gray',
        alpha=0.7
    )
    ax.set_ylim(0, 1.1)
    
    # 绘制数据 (移除fill填充，只保留线条)
    for i in range(len(data_subset)):
        name = data_subset['Name'].iloc[i]
        values = data_subset.iloc[i, 1:].tolist()
        values += values[:1]
        ax.plot(angles, values, label=name, 
                color=name_to_color[name],  # 使用预定义颜色
                alpha=0.9, linewidth=1.2)  # 加粗线条宽度
        ax.fill(angles, values, color=name_to_color[name], alpha=0.1)# 填充透明度
        
    # 绘制平均值
    # mean_values = data_subset.iloc[:, 1:].mean().tolist()
    # mean_values += mean_values[:1]
    # ax.plot(angles, mean_values, label='Mean', 
    #         linestyle='--', linewidth=1.5, color='black')

    # 设置标签和标题
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=8, pad=15)
    
    # 调整图例
    legend = ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.05),
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )
    legend.get_frame().set_linewidth(0.4)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(
        os.path.join(save_dir, filename),
        dpi=300,
        facecolor='white',
        edgecolor='none',
        bbox_inches='tight',
        transparent=False
    )
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