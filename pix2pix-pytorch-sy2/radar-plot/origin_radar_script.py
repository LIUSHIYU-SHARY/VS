
import originpro as op
import pandas as pd
import numpy as np
import os

# 原始数据
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

# 指标归一化
normalized_df = df.copy()
normalized_df['LPIPS'] = 1 - normalized_df['LPIPS'] / normalized_df['LPIPS'].max()
normalized_df['FID'] = 1 - normalized_df['FID'] / normalized_df['FID'].max()
for col in ['PSNR (dB)', 'SSIM', 'DICE', 'Pearson']:
    normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# 创建 Origin 工作表
op.set_show(True)
wks = op.new_sheet("w", lname="RadarData")
wks.from_df(normalized_df)

# 可选：创建雷达图模板或使用自定义函数绘图
# 当前 Origin 无官方雷达图模板，建议以 Line + Polar 或 Radar Spider 图模板支持

# 如果你有预设的 radar.opju 项目文件，也可以用以下方式打开：
# op.open(file='path_to_your_radar_template.opju')
