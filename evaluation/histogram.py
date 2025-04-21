import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from PIL import Image

def load_image_as_array(image_path):
    """加载图像并转换为灰度数组"""
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    return np.array(image)

def plot_joint_histogram(image1, image2):
    """绘制两个图像的联合直方图"""
    # 检查图像尺寸是否相同，不同则裁剪
    min_shape = min(image1.shape, image2.shape)
    image1 = image1[:min_shape[0], :min_shape[1]]
    image2 = image2[:min_shape[0], :min_shape[1]]
    
    # 计算联合直方图
    joint_histogram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=256, range=[[0, 256], [0, 256]])
    
    # 绘制二维直方图
    plt.figure(figsize=(8, 6))
    plt.imshow(joint_histogram, norm=LogNorm(vmin=1, vmax=100), origin='lower', interpolation='nearest', aspect='auto', cmap='plasma', extent=[0, 256, 0, 256])
    plt.colorbar(label='Counts')
    plt.title("Joint Histogram of Two Images")
    plt.xlabel("Pixel Intensity of Image 1")
    plt.ylabel("Pixel Intensity of Image 2")
    plt.show()

# 图像路径
image_path1 = '..\\images\\100_real.jpg'
image_path2 = '..\\images\\100_fake.jpg'

# 加载图像并转换为数组
image1 = load_image_as_array(image_path1)
image2 = load_image_as_array(image_path2)

# 绘制两个图像的联合直方图
plot_joint_histogram(img1, img2)