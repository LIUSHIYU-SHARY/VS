import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stardist.plot import render_label

# --------- 参数配置 ----------
# real_folder = "/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase-512/test/b"
# virtual_folder = "/home/yuqi/virtual_staining/project_code/pix2pix-pytorch-sy/Nucleus_result/0405_512_20x_phase"
real_folder = "/home/yuqi/virtual_staining/tmp_data/Nuclear-classification/20x/20x-phase-1024/test/b"
virtual_folder = "/home/yuqi/virtual_staining/project_code/pix2pix-pytorch-sy/Nucleus_result/0409_1024_20x_phase"
save_image_dir = "/home/yuqi/virtual_staining/project_code/count_nucleus-sy/count_img/0409_1024_20x_phase_final"
save_excel_path = "/home/yuqi/virtual_staining/project_code/count_nucleus-sy/count_log/0409_1024_20x_phase_final.xlsx"

os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)

# --------- 加载模型 ----------
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# --------- 计数函数 ----------
# def count_cells(img_path, min_size=150, prob_thresh=0.25, nms_thresh=0.75):
def count_cells(img_path, min_size=400, prob_thresh=0.25, nms_thresh=0.75):
    img = Image.open(img_path).convert("L")
    img_array = np.array(img)
    normalized = normalize(img_array)
    labels, _ = model.predict_instances(normalized, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    unique_labels = np.unique(labels)[1:]
    areas = ndimage.sum(np.ones_like(labels), labels, index=unique_labels)

    filtered = labels.copy()
    for label, area in zip(unique_labels, areas):
        if area < min_size:
            filtered[labels == label] = 0

    return img_array, filtered, len(np.unique(filtered)) - 1

# --------- 文件检查 ----------
real_files = set(f for f in os.listdir(real_folder) if f.endswith((".png", ".jpg", ".jpeg", ".tif")))
virtual_files = set(f for f in os.listdir(virtual_folder) if f.endswith((".png", ".jpg", ".jpeg", ".tif")))

only_in_real = sorted(real_files - virtual_files)
only_in_virtual = sorted(virtual_files - real_files)

if only_in_real:
    print("⚠️ 以下文件只存在于 Real 文件夹中，但 Virtual 中没有：")
    for f in only_in_real:
        print("  -", f)

if only_in_virtual:
    print("⚠️ 以下文件只存在于 Virtual 文件夹中，但 Real 中没有：")
    for f in only_in_virtual:
        print("  -", f)

common_files = sorted(real_files & virtual_files)
print(f"\n✅ 共发现 {len(common_files)} 对文件用于分析")

# --------- 主循环 ----------
results = []

for filename in common_files:
    input_path = os.path.join(real_folder, filename)
    output_path = os.path.join(virtual_folder, filename)

    img_input, input_labels, input_count = count_cells(input_path)
    img_output, output_labels, output_count = count_cells(output_path)

    count_diff = input_count - output_count

    results.append({
        "Filename": filename,
        "Fluorescent Image Count": input_count,
        "Virtual Staining Image Count": output_count,
        "Count Difference": count_diff
    })

    overlay_input = render_label(input_labels, img=img_input)
    overlay_output = render_label(output_labels, img=img_output)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(img_input, cmap='gray')
    axes[0, 0].set_title("Fluorescent Image Original")
    axes[1, 0].imshow(overlay_input)
    axes[1, 0].set_title(f"Fluorescent Image Count: {input_count}")
    axes[0, 1].imshow(img_output, cmap='gray')
    axes[0, 1].set_title("Virtual Staining Image Original")
    axes[1, 1].imshow(overlay_output)
    axes[1, 1].set_title(f"Virtual Staining Image Count: {output_count}")

    for ax in axes.flatten():
        ax.axis('off')

    fig.suptitle(f"{filename}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_image_dir, f"compare_{filename}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 保存四宫格图像: {save_path}")

# --------- 保存 Excel ---------
df = pd.DataFrame(results)
df.to_excel(save_excel_path, index=False)
print(f"\n📄 所有统计结果已保存到: {save_excel_path}")
