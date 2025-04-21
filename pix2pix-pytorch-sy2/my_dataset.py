from os import listdir
from os.path import join
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # 确定是哪个图像和对应的子块编号
        image_index = index // 4  # 图像编号
        block_index = index % 4  # 子块编号 (0, 1, 2, 3 分别表示左上、右上、左下、右下)

        # 加载对应的图像
        a = Image.open(join(self.a_path, self.image_filenames[image_index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[image_index])).convert('RGB')

        # 确保图像大小为 512x512
        a = a.resize((512, 512), Image.BICUBIC)
        b = b.resize((512, 512), Image.BICUBIC)

        # 根据子块编号裁剪图像
        if block_index == 0:  # 左上
            crop_box = (0, 0, 256, 256)
        elif block_index == 1:  # 右上
            crop_box = (256, 0, 512, 256)
        elif block_index == 2:  # 左下
            crop_box = (0, 256, 256, 512)
        elif block_index == 3:  # 右下
            crop_box = (256, 256, 512, 512)

        a = a.crop(crop_box)
        b = b.crop(crop_box)

        # 转换为张量并标准化
        a = self.transform(a)
        b = self.transform(b)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        # 每个图像有 4 个子块
        return len(self.image_filenames) * 4