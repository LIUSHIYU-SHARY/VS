import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import pandas as pd
from timm import create_model
import shutil



GOOD_DIR = "/home/yuqi/virtual_staining/tmp_data/classifier/class_0"
BAD_DIR = "/home/yuqi/virtual_staining/tmp_data/classifier/class_1"
TEST_DIR = "/home/yuqi/virtual_staining/tmp_data/Nuclear-classification/20x/20x-phase/input"
GOOD_OUTPUT_DIR = "/home/yuqi/virtual_staining/tmp_data/Nuclear-classification/20x/20x-phase/input_goodimages"
# TEST_DIR = "/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase-1024/test/a"
# GOOD_OUTPUT_DIR = "/home/yuqi/virtual_staining/tmp_data/20x-phase-1024/good_nuclei_images"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# FEATURE_DIM = 2048  # ResNet50 avgpool输出维度


# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
results = []

#FEATURE_DIM = 1024  # Swin-T的特征维度，根据具体模型调整

# 初始化特征提取模型
model = create_model('swin_large_patch4_window7_224', pretrained=True)
model.head = torch.nn.Identity()
model = model.to(DEVICE).eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    """提取图像特征向量（支持TIFF大图）"""
    img = Image.open(img_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(tensor)
        features = features.reshape(features.size(0), -1)       # swin_tansformer new
    return features.squeeze().cpu().numpy()

def calculate_centroid(directory):
    """计算指定目录下所有图片的特征质心"""
    features = []
    for fname in os.listdir(directory):
        if fname.lower().endswith('.tif'):
            img_path = os.path.join(directory, fname)
            features.append(extract_features(img_path))
    return np.mean(features, axis=0)

os.makedirs(GOOD_OUTPUT_DIR, exist_ok=True)
# 计算两类质心
good_centroid = calculate_centroid(GOOD_DIR)
bad_centroid = calculate_centroid(BAD_DIR)

def classify_image(img_path):
    """
    图像质量分类函数
    返回：("Good"/"Bad", 置信度)
    """
    # 特征提取
    feature = extract_features(img_path)
    
    # 计算余弦相似度
    sim_good = np.dot(feature, good_centroid) / (
        np.linalg.norm(feature) * np.linalg.norm(good_centroid))
    sim_bad = np.dot(feature, bad_centroid) / (
        np.linalg.norm(feature) * np.linalg.norm(bad_centroid))
    
    # 计算置信度
    confidence = abs(sim_good - sim_bad)
    if sim_good > sim_bad:
        return "Good", confidence
    else:
        return "Bad", confidence

# 执行分类
for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith('.tif'):
        continue
    
    img_path = os.path.join(TEST_DIR, img_name)
    class_name, confidence = classify_image(img_path)
    results.append({
                    'Image_Name': img_name,
                    'Predicted_Class': class_name,
                    'Confidence_score': f"{confidence:.4f}",
                })
    
    if class_name == "Good":
        shutil.copy2(img_path, os.path.join(GOOD_OUTPUT_DIR, img_name))
        
    print(f"{img_name}:")
    print(f"  Classification: {class_name}")
    print(f"  Confidence: {confidence:.2f}")
    print("-" * 40)
    
df = pd.DataFrame(results)
csv_path = '/home/yuqi/virtual_staining/tmp_data/Nuclear-classification/20x/20x-phase/prediction_20x_512_phase.csv'
df.to_csv(csv_path, index=False)

print(f"\n预测结果已保存到：{csv_path}")