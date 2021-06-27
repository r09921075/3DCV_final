
from tqdm import tqdm
import glob
import numpy as np
import cv2 as cv
import math


 
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ssim(x, y):
	std_x = np.std(x)
	std_y = np.std(y)
	L=255
	c1=c2=1/np.sqrt(L)
	mean_x = np.mean(x)
	mean_y = np.mean(y)
	var_xy = np.mean((x-mean_x)*(y-mean_y))
	ssim = (2*mean_x*mean_y+(c1*L)**2)*(2*var_xy+(c2*L)**2)/(mean_x**2+mean_y**2+(c1*L)**2)/(std_x**2+std_y**2+(c2*L)**2)
	return ssim

def _main():
	path = "data/out_img/slomo/*.png" 
	path1 = "data/resource_img/car/*.png"
	frame_list = sorted(glob.glob(path),key=len)
	frame1_list = sorted(glob.glob(path1),key=len)
	length = len(frame_list)
	SSIM=0
	PSNR=0
	for img,img1 in tqdm(zip(frame_list,frame1_list)):
		t1 = cv.imread(img)
		t2 = cv.imread(img1)
		SSIM += ssim(t1,t2)
		PSNR += psnr(t1,t2)
	print(SSIM/length)
	print(PSNR/length)



		


if __name__ == '__main__':
    _main()