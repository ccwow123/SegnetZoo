import glob
import os
import sys
sys.path.insert(0, '../')

import time
import json
import argparse
import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import namedtuple
from itertools import product
from tqdm import tqdm, trange
import re
from utils import transforms as T
from utils.my_dataset import VOCSegmentation
# 调试板初始化
def palette_init():
    palette_path = r'palette_utils/palette.json'
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    return pallette
# 创建输出路径文件夹
def create_save_path(weights_path):

    model_name = weights_path.split('/')[-2].split('-')[-1]
    save_path = os.path.join('output', model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def get_transform(base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    class SegmentationPresetTest:
        def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            self.transforms = T.Compose([
                T.Resize(base_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

        def __call__(self, img, target):
            return self.transforms(img, target)
    return SegmentationPresetTest(base_size, mean, std)
def main(args):
    # 初始化
    palette = palette_init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path=create_save_path(args.weights_path)
    pre_dataset= VOCSegmentation(args.data_path,
                                  year="2007",
                                  transforms=get_transform(args.img_size),
                                  txt_name="test.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    parser.add_argument("--weights_path", default=r'logs/04-05_16-12-15-Unet_c2f/best_model.pth', type=str, help="权重路径")
    parser.add_argument("--data_path", default=r'D:\Files\_datasets\Dataset-reference\VOC_E_skew', help="VOCdevkit 路径")
    parser.add_argument("--num-classes", default=2, type=int,help="分类总数")
    parser.add_argument("--img-size", default=512, type=int,help="图片缩放大小")
    parser.add_argument("--method", default="fusion",  choices=["fusion", "mask", "contours"], help="输出方式")
    # 其他
    parser.add_argument("--label", default="End skew", type=str, help="contours方式下的标签")

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
