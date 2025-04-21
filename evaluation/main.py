import cv2
import numpy as np

import _utils_calculate_psnr_ssim as util

# 加载需要计算的两张图片, opencv读取方式是<numpy.ndarray>, HWC, BGR, <numpy.uint8>[0, 255]
img1 = cv2.imread('../pix2pix-pytorch/result/108.jpg')
img2 = cv2.imread('../autodl-tmp/easy/original/test/b/108.jpg')
#img2 = cv2.imread('../images/cycle_gan_100_fake.jpg')



# 设定图像边缘裁剪像素
crop_border = 0

# 计算psnr与ssim (输入图片的格式需要是ndarray)
psnr = util.calculate_psnr(img1, img2, crop_border, test_y_channel=True)
ssim = util.calculate_ssim(img1, img2, crop_border, test_y_channel=True)
#fsim=util.calculate_fsim(img1,img2)
lpips = util.calculate_lpips(img1, img2)
dice=util.dice_coefficient(img1, img2)
pearson_correlation=util.pearson_correlation(img1,img2)


#print(f'PSNR: {psnr:.6f}dB\nSSIM: {ssim:.6f}\nFSIM: {fsim:.6f}\nLPIPS: {lpips:.6f}')
print(f'PSNR: {psnr:.6f}dB\nSSIM: {ssim:.6f}\ndice:{dice:.6f}\npearson_correlation{pearson_correlation:.6f}\nLPIPS: {lpips:.6f}')
util.plot_joint_histogram(img1, img2)
