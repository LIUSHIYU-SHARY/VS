
import pandas as pd
import matplotlib.pyplot as plt

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
    normalized_df[col] = (normalized_df[col] - df[col].min()) / (df[col].max() - df[col].min())

# 雷达图绘制
categories = ['PSNR (dB)', 'SSIM', 'LPIPS', 'DICE', 'Pearson', 'FID']
angles = [n / float(len(categories)) * 2 * 3.1416 for n in range(len(categories))]
angles += angles[:1]

plt.figure(figsize=(10, 7))
for i in range(len(df)):
    values = normalized_df.iloc[i, 1:].tolist()
    values += values[:1]
    plt.polar(angles, values, label=df['Name'][i], alpha=0.6)

plt.xticks(angles[:-1], categories)
plt.title("Virtual Staining Performance Comparison", size=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("Virtual Staining Performance Comparison.png")
plt.show()
