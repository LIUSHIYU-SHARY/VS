# from os import listdir
# from os.path import join
# import random

# from PIL import Image
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms

# from utils import is_image_file, load_img


# class DatasetFromFolder(data.Dataset):
#     def __init__(self, image_dir, direction):
#         super(DatasetFromFolder, self).__init__()
#         self.direction = direction
#         self.a_path = join(image_dir, "a")
#         self.b_path = join(image_dir, "b")
#         self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

#         transform_list = [transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

#         self.transform = transforms.Compose(transform_list)

#     def __getitem__(self, index):
#         a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
#         b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
#         # 修改resize尺寸为542x542
#         a = a.resize((542, 542), Image.BICUBIC)
#         b = b.resize((542, 542), Image.BICUBIC)
#         a = transforms.ToTensor()(a)
#         b = transforms.ToTensor()(b)
#         # 修改随机裁剪的偏移范围
#         w_offset = random.randint(0, max(0, 542 - 512 - 1))
#         h_offset = random.randint(0, max(0, 542 - 512 - 1))
    
#         # 修改裁剪尺寸为512x512
#         a = a[:, h_offset:h_offset + 512, w_offset:w_offset + 512]
#         b = b[:, h_offset:h_offset + 512, w_offset:w_offset + 512]
    
#         a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
#         b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

#         if random.random() < 0.5:
#             idx = [i for i in range(a.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             a = a.index_select(2, idx)
#             b = b.index_select(2, idx)

#         if self.direction == "a2b":
#             return a, b
#         else:
#             return b, a

#     def __len__(self):
#         return len(self.image_filenames)


from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


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
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        # 修改resize尺寸为542x542
        a = a.resize((1054, 1054), Image.BICUBIC)
        b = b.resize((1054, 1054), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        # 修改随机裁剪的偏移范围
        w_offset = random.randint(0, max(0, 1054 - 1024 - 1))
        h_offset = random.randint(0, max(0, 1054 - 1024 - 1))
    
        # 修改裁剪尺寸为512x512
        a = a[:, h_offset:h_offset + 1024, w_offset:w_offset + 1024]
        b = b[:, h_offset:h_offset + 1024, w_offset:w_offset + 1024]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)