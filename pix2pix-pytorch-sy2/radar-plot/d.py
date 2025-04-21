import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# 模型数据
data = {
    "Model": ["CycleGAN baseline", "Pix2pix baseline", "UGATIT", "Pix2pix turbo", "Our model"],
    "PSNR (dB)": [16.277223, 29.172701, 23.091845, 30.341801, 33.395420],
    "SSIM": [0.548495, 0.969502, 0.933814, 0.97487, 0.963976],
    "LPIPS": [0.423364, 0.108565, 0.223881, 0.117987, 0.196629],
    "Dice": [0.115297, 0.943085, 0.913908, 0.946142, 0.81512],
    "Pearson": [0.1696, 0.918376, 0.75854, 0.917381, 0.92001],
    "FID": [100.899724, 29.412839, 35.104237, 32.621197, 75.99919]
}
df = pd.DataFrame(data)

# 配置
metrics = ["PSNR (dB)", "SSIM", "LPIPS", "Dice", "Pearson", "FID"]
prefer_higher = ["PSNR (dB)", "SSIM", "Dice", "Pearson"]
prefer_lower = ["LPIPS", "FID"]
grayscale_colors = ['lightgray', 'gray', 'dimgray', 'darkgray', 'black']

# 处理单位
titles_with_units = []
for metric in metrics:
    if df[metric].max() <= 1:
        titles_with_units.append(f"{metric} (%)")
    else:
        titles_with_units.append(metric)

# 绘图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

for idx, (ax, metric, title) in enumerate(zip(axs.flat, metrics, titles_with_units)):
    values = df[metric]
    normalized = values / values.max() * 100

    bars = ax.bar(range(len(values)), normalized, color=grayscale_colors)

    for i, (bar, val) in enumerate(zip(bars, values)):
        label = f'{val*100:.1f}%' if val < 1 else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontsize=10)

    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xticks(range(len(df["Model"])))
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylim(0, 110)
    ax.grid(False)

# 图例（去掉斜线说明）
gray_patches = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(grayscale_colors, df["Model"])
]
plt.legend(handles=gray_patches, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

# 标题与注释
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9)
# plt.suptitle("Comparison of Model Performance (CycleGAN, Pix2Pix, UGATIT, etc.)", fontsize=16, y=1.05)
plt.suptitle("Comparison of Model Performance (Our model, CycleGAN, Pix2Pix, etc.)", fontsize=16, y=1.05)

note_text = """Notes:
1. Higher values are better for PSNR(dB), SSIM(%), Dice(%), Pearson(%)
2. Lower values are better for LPIPS(%), FID"""
plt.figtext(0.1, -0.05, note_text, horizontalalignment='left', fontsize=12)

# 显示和保存
plt.show()
plt.savefig("new_5_models.png", dpi=600, bbox_inches='tight')
