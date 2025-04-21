# python test.py --dataset ../../tmp_data/Minigut
# python test.py --dataset ../../tmp_data/in_silico

from __future__ import print_function
import argparse
import os
import torch
import torchvision.transforms as transforms
# from networks import define_G  # Adjust this import according to your actual model definition file
from my_network_test import define_G
from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:1" if opt.cuda else "cpu")

# model_path = "./result/my_checkpoints/netG_model_epoch_100.pth"
#model_path = "../data/my_checkpoint/pix2pix/netG_model_epoch_20.pth"
#model_path = "/home/yuqi/virtual_staining/output/pix2pix/Nucleus/old_minigut_checkpoint/netG_model_epoch_200.pth"
model_path = "/home/yuqi/virtual_staining/output/pix2pix/Nucleus/0318_insilico_checkpoint/netG_model_epoch_200.pth"

# Define the model architecture
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
state_dict = torch.load(model_path, map_location=device,weights_only=True)
net_g.load_state_dict(state_dict)
net_g.to(device)
net_g.eval()  # Set the model to evaluation mode if not training

if opt.direction == "a2b":
    image_dir = "{}/test/a/".format(opt.dataset)
else:
    image_dir = "{}/test/b/".format(opt.dataset)

print(image_dir)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)
# result_dir = "freeze_pretrained_100"
result_dir = "Nucleus_result/new_insilico"
os.makedirs(result_dir, exist_ok=True) 

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img(out_img, "{}/{}".format(result_dir, image_name))