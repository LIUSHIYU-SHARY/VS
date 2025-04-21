import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据准备
data = {
    "Metric": ["PSNR(↑)", "SSIM(↑)", "LPIPS(↓)", "Dice(↑)", "Pearson(↑)", "FID(↓)"],
    "Without Transfer": [32.532528, 0.96305, 0.233283, 0.787785, 0.901501, 67.924527],
    "With Transfer": [33.395420, 0.963976, 0.196629, 0.81512, 0.92001, 75.99919]
}

df = pd.DataFrame(data)
x = np.arange(len(df["Metric"]))  # 横轴坐标位置
width = 0.35  # 柱子宽度

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, df["Without Transfer"], width, label="Without Transfer", color="#1a6fdf")
bars2 = ax.bar(x + width/2, df["With Transfer"], width, label="With Transfer", color="#b177de")

# 设置标签和标题
# ax.set_xlabel("Metric")
ax.set_ylabel("Value")
ax.set_title("Comparison Of Evaluation Metrics With And Without Transfer Learning in Minigut Dataset")
ax.set_xticks(x)
ax.set_xticklabels(df["Metric"])
ax.legend()

# 添加柱子顶部数值标签
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 垂直偏移
                textcoords="offset points",
                ha='center', va='bottom')

# 设置黑色坐标轴边框
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# 去除背景网格线
ax.grid(False)

# 保存图像（可选）
plt.tight_layout()
plt.savefig("comparison_metrics_transfer_learning.png", dpi=300)
plt.show()
