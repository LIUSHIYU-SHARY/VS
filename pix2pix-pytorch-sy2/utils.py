import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #img = img.resize((256, 256), Image.BICUBIC)
    return img

#
#def save_img(image_tensor, filename):
#    image_numpy = image_tensor.float().numpy()
#    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#    image_numpy = image_numpy.clip(0, 255)
#    image_numpy = image_numpy.astype(np.uint8)
#    image_pil = Image.fromarray(image_numpy)
#    image_pil.save(filename)
#    print("Image saved as {}".format(filename))

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # 应用对比度增强
    image_numpy = (image_numpy + 1) / 2.0  # 转换到[0,1]范围
    image_numpy = np.power(image_numpy, 0.9)  # 微调对比度
    
    image_numpy = image_numpy * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    
    # 保存为更高质量的图像格式
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename, quality=95)  # 增加保存质量
