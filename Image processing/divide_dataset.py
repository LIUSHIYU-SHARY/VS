import os
import shutil
import random

def split_images(input_folder, output_folder, train_folder, test_folder, total_images, train_ratio=0.8):
    # 获取 input 文件夹中的所有图片
    images = sorted(os.listdir(input_folder))
    
    # 确保图像对齐
    output_images = sorted(os.listdir(output_folder))
    assert images == output_images, "Input 和 Output 文件夹的图片名称不匹配！"
    
    # 确保提供的总数和实际图片数匹配
    assert len(images) == total_images, f"提供的数据总数 {total_images} 与实际文件数 {len(images)} 不匹配！"
    
    # 随机打乱顺序
    random.shuffle(images)
    
    # 计算训练集数量
    train_size = int(total_images * train_ratio)
    test_size = total_images - train_size

    # 划分数据集
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    # 创建训练集和测试集文件夹
    for folder in [train_folder, test_folder]:
        os.makedirs(os.path.join(folder, 'a'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'b'), exist_ok=True)
    
    # 复制训练集图片
    for img in train_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(train_folder, 'a', img))
        shutil.copy(os.path.join(output_folder, img), os.path.join(train_folder, 'b', img))
    
    # 复制测试集图片
    for img in test_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(test_folder, 'a', img))
        shutil.copy(os.path.join(output_folder, img), os.path.join(test_folder, 'b', img))
    
    print(f"数据集划分完成：训练集 {train_size} 对，测试集 {test_size} 对")

def split_images_for_magnification(magnification, total_images):
    input_path = os.path.join(magnification, "input")
    output_path = os.path.join(magnification, "output1")
    train_path = os.path.join(magnification, "train")
    test_path = os.path.join(magnification, "test")
    
    split_images(input_path, output_path, train_path, test_path, total_images)

# 使用示例（总数据量 72 张）
split_images_for_magnification("20x/20x-bright", 288)
split_images_for_magnification("20x/20x-phase", 180)
split_images_for_magnification("40x/40x-bright", 270)

