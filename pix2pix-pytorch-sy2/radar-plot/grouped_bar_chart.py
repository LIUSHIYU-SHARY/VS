
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据准备
data = {
    "Error Group": ["<5%", "5%-10%", "10%-20%", "20%-30%", ">=30%"],
    # "512x512": [22.2, 11.1, 50, 8.3, 8.3],
    # "1024x1024": [41.7, 22.2, 16.7, 11.1, 8.3]
    "512x512": [46.9, 18.8, 31.3, 0, 3.1],
    "1024x1024": [51.3, 37.5, 9.4, 0, 0]
}

df = pd.DataFrame(data)
x = np.arange(len(df["Error Group"]))  # 横轴坐标
width = 0.35  # 柱子宽度

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width/2, df["512x512"], width, label="512x512")
# bars2 = ax.bar(x + width/2, df["1024x1024"], width, label="1024x1024")
bars1 = ax.bar(x - width/2, df["512x512"], width, label="512x512", color="#1a6fdf")
bars2 = ax.bar(x + width/2, df["1024x1024"], width, label="1024x1024", color="#b177de")


# 标签与图例
ax.set_xlabel("Error Group")
ax.set_ylabel("Value (%)")
ax.set_title("Comparison of Error Distribution Between Resolutions ( Classifier-Filtered)")
ax.set_xticks(x)
ax.set_xticklabels(df["Error Group"])
ax.legend()

# 添加柱子上的数值标签
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 垂直偏移
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig("error_distribution_comparison(Classifier-Filtered).png", dpi=300)
plt.show()
