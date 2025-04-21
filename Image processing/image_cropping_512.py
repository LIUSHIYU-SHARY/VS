# 裁剪训练集 & 复制测试集

import os
import shutil
from PIL import Image

def crop_paired_images(imgA_path, imgB_path, output_dirA, output_dirB, crop_size=512, pad=24):
    """
    切割配对图像，并分别保存到两个文件夹中
    :param imgA_path: 图像A路径
    :param imgB_path: 图像B路径
    :param output_dirA: 图像A的输出目录
    :param output_dirB: 图像B的输出目录
    :param crop_size: 切割尺寸 默认512
    :param pad: 边缘留空 默认24
    """
    # 确保输出目录存在
    os.makedirs(output_dirA, exist_ok=True)
    os.makedirs(output_dirB, exist_ok=True)

    # 打开配对图像
    imgA = Image.open(imgA_path)
    imgB = Image.open(imgB_path)

    # 验证图像尺寸
    if imgA.size != (2000, 2000) or imgB.size != (2000, 2000):
        raise ValueError("输入图像尺寸必须为2000x2000像素")

    # 计算滑动步长
    step = (2000 - 2 * pad - crop_size) // 7  # 7步切割8次

    # 生成切割坐标
    positions = [(x, y) for y in range(pad, 2000 - pad - crop_size + 1, step)
                 for x in range(pad, 2000 - pad - crop_size + 1, step)]

    # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(imgA_path))[0][:25]

    # 执行切割操作
    for idx, (left, top) in enumerate(positions):
        box = (left, top, left + crop_size, top + crop_size)

        # 切割并保存图像A
        cropA = imgA.crop(box)
        cropA.save(os.path.join(output_dirA, f"{base_name}_{idx:02d}.png"))

        # 切割并保存图像B
        cropB = imgB.crop(box)
        cropB.save(os.path.join(output_dirB, f"{base_name}_{idx:02d}.png"))

def process_folders(folderA, folderB, output_dirA, output_dirB, crop_size=512, pad=24):
    """
    处理两个文件夹中的所有配对图像，并分别保存到两个输出文件夹中
    :param folderA: 文件夹A路径
    :param folderB: 文件夹B路径
    :param output_dirA: 图像A的输出目录
    :param output_dirB: 图像B的输出目录
    :param crop_size: 切割尺寸 默认512
    :param pad: 边缘留空 默认24
    """
    # 获取两个文件夹中的文件列表
    filesA = sorted(os.listdir(folderA))
    filesB = sorted(os.listdir(folderB))

    # 检查文件数量是否一致
    if len(filesA) != len(filesB):
        raise ValueError("两个文件夹中的文件数量不一致")

    # 遍历所有文件
    for fileA, fileB in zip(filesA, filesB):
        if fileA != fileB:
            raise ValueError(f"文件名不匹配: {fileA} 和 {fileB}")

        # 构建完整路径
        imgA_path = os.path.join(folderA, fileA)
        imgB_path = os.path.join(folderB, fileB)

        # 处理当前配对图像
        print(f"正在处理: {fileA} 和 {fileB}")
        crop_paired_images(imgA_path, imgB_path, output_dirA, output_dirB, crop_size, pad)

def copy_test_folder(original_path, magnification_cropping_path):
    """
    复制 original_path 下的 test 目录到 magnification_cropping_path 下
    """
    source_test_path = os.path.join(original_path, "test")
    destination_test_path = os.path.join(magnification_cropping_path, "test")

    # 确保目标目录存在
    os.makedirs(destination_test_path, exist_ok=True)

    # 复制 test 目录下的所有文件和子目录
    if os.path.exists(source_test_path):
        shutil.copytree(source_test_path, destination_test_path, dirs_exist_ok=True)
        print(f"✅ 成功复制 {source_test_path} -> {destination_test_path}")
    else:
        print(f"⚠️ 警告：{source_test_path} 目录不存在，跳过复制")

def magnification_cropping(original_path, magnification_cropping_path):
    """
    对训练集数据进行裁剪，并复制测试集数据
    """
    train_input_path = os.path.join(original_path, "train", "a")
    train_output_path = os.path.join(original_path, "train", "b")
    
    output_dirA = os.path.join(magnification_cropping_path, "train", "a")
    output_dirB = os.path.join(magnification_cropping_path, "train", "b")

    print(f"🔹 开始裁剪训练集: {train_input_path} -> {output_dirA}")
    process_folders(train_input_path, train_output_path, output_dirA, output_dirB)

    # 复制测试集
    copy_test_folder(original_path, magnification_cropping_path)

# if __name__ == "__main__":
#     #修改这两处函数即可
#     # 原始数据集路径，包含 train 和 test 文件夹，可以修改为20x/20x-phase，40x/40x-phase等等
#     original_path = "20x/20x-bright" 
#     # 裁剪后数据集路径 
#     magnification_cropping_path = "20x/20x-bright-512" 

#     # 处理训练集 & 复制测试集
#     magnification_cropping(original_path, magnification_cropping_path)

if __name__ == "__main__":
    # 定义原始数据集和裁剪后数据集的映射
    dataset_mapping = {
        "20x/20x-phase": "20x/20x-phase-512",
        "20x/20x-bright": "20x/20x-bright-512",
        "40x/40x-bright": "40x/40x-bright-512"
    }

    # 处理每个数据集
    for original_path, magnification_cropping_path in dataset_mapping.items():
        print(f"📌 正在处理: {original_path} -> {magnification_cropping_path}")
        magnification_cropping(original_path, magnification_cropping_path)

    print("🎉 所有数据集处理完成！")