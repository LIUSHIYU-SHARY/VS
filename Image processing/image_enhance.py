import cv2
import numpy as np
import os

# 定义输入和输出文件夹的映射 (HashMap)
folders = {
    "20x/20x-phase/output0": "20x/20x-phase/output1",
    "20x/20x-bright/output0": "20x/20x-bright/output1",
    "40x/40x-bright/output0": "40x/40x-bright/output1"
}

# 原始卷积核
custom_kernel = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 50, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
], dtype=np.float32)

# 归一化卷积核（避免亮度溢出）
if custom_kernel.sum() != 0:
    custom_kernel /= custom_kernel.sum()

# 遍历所有的 input_folder -> output_folder
for input_folder, output_folder in folders.items():
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在
    print(f"📂 正在处理: {input_folder} -> {output_folder}")

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    if not image_files:
        print(f"⚠️ 警告: {input_folder} 为空，没有可处理的图像。")
        continue

    for img_name in image_files:
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ 无法读取 {img_name}，跳过...")
            continue

        # 应用卷积（使用 ImageJ 风格的边界处理）
        convolved_img = cv2.filter2D(img.astype(np.float32), -1, custom_kernel, borderType=cv2.BORDER_REFLECT)

        # 转换回 uint8 避免超出范围
        convolved_img = np.clip(convolved_img, 0, 255).astype(np.uint8)

        # 确保写入成功
        success = cv2.imwrite(output_path, convolved_img)
        if success:
            print(f"✅ 处理完成: {img_name} -> {output_path}")
        else:
            print(f"❌ 失败: 无法保存 {output_path}")

print("🎉 所有批量处理完成！")
