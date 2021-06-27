import numpy as np
import cv2 as cv
import argparse

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# parser.add_argument('data\slow_traffic_small.mp4', type=str, help='path to image file')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.image)
path1 = 'data/need_img/car/'
path2 = 'data/out_img/opencv/car/'
path1=path2
number=0
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.createBackgroundSubtractorMOG2()

h, w = cv.imread(path1+str(number)+'.png',0).shape[:2]
dis = 2
while(1):
    
    now_img = cv.imread(path1+str(number)+'.png')
    
    cv.imwrite(path2+str(number)+'.png',now_img)
    if number>=912:break

    next_img = cv.imread(path1+str(number+dis)+'.png')
    
    
    now_img1 = cv.cvtColor(now_img,cv.COLOR_BGR2GRAY)
    next_img = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    
    
    # now_img1 = fgbg.apply(now_img1)
    # now_img1 = cv.morphologyEx(now_img1, cv.MORPH_OPEN, kernel)

    # next_img = fgbg.apply(next_img)
    # next_img = cv.morphologyEx(next_img, cv.MORPH_OPEN, kernel)
    flow = -cv.calcOpticalFlowFarneback(now_img1,next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)/2      #計算flow /2從中間插入

    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    
    
    mid_Img = cv.remap(now_img, flow, None, cv.INTER_LINEAR)  
    cv.imshow('123',mid_Img)
    # mid_Img=flow_warp(now_img1, flow, interp_mode='bilinear', padding_mode='zeros')
    cv.imwrite(path2+str(int(number+dis/2))+'.png',mid_Img)
    number+=dis
    cv.waitKey(10)

