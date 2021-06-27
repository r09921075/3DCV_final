import sys
from torch import tensor
sys.path.append('core')
from torch.nn import functional as F
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from core.core_model import RAFT
from core.utils import InputPadder
import matplotlib.pyplot as plt
from tqdm import tqdm
DEVICE = 'cuda'

def load_image(imfile):
    img = cv2.imread(imfile)
    img1 = img
    img = img.astype(np.uint8)

    
    
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    
    return img[None].to(DEVICE),img1

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    path1 = 'data/need_img/ssg/*.png'
    path2 = 'data/out_img/raft/f1/'
    path3 = 'data/out_img/raft/f1/*.png'
    # path1=path3
    distance = 2
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        frame_list = sorted(glob.glob(path1),key=len)
        # print(frame_list)
        # print(frame_list)
        
        i=0
        for j in tqdm(range(len(frame_list))):
            image1,image1_32 = load_image(frame_list[j])
            cv2.imwrite(path2+str(i)+'.png',image1_32)
            image2,image2_32 = load_image(frame_list[j+1])
            # print(imfile1.shape)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            
            
            flow_low, flow_up = model(image1, image2, iters=1, test_mode=True)
            flow_low, flow_up_back = model(image2, image1, iters=1, test_mode=True)

            img = image1[0].permute(1,2,0).cpu().numpy()
            flow = -flow_up[0].permute(1,2,0).cpu().numpy()
            h, w = img.shape[:2]
            
            flow[:,:,0] += np.arange(w)
            flow[:,:,1] += np.arange(h)[:,np.newaxis]
            mid_Img = cv2.remap(image1_32, flow, None, cv2.INTER_LINEAR)
            # cv2.imwrite(path2+str(i)+'.png',image1_32)
            cv2.imwrite(path2+str(int(i+0.5*distance))+'.png',mid_Img)
            i+=distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--path', help="dataset for evaluation")
    # parser.add_argument('--outpath', help="dataset for evaluation")
    # parser.add_argument('--framedis', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    demo(args)