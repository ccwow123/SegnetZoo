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
from utils.img_process import contours_process_nolabel
# 调试板初始化
def palette_init():
    palette_path = r'palette_utils/palette.json'
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    return pallette,pallette_dict
# 创建输出路径文件夹
def create_save_path(weights_path):

    model_name = weights_path.split('/')[-2].split('-')[-1]
    save_path = os.path.join('output', model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

# def get_transform(base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     class SegmentationPresetTest:
#         def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#             self.transforms = T.Compose([
#                 T.Resize(base_size),
#                 T.ToTensor(),
#                 T.Normalize(mean=mean, std=std),
#             ])
#
#         def __call__(self, img, target):
#             return self.transforms(img, target)
#     return SegmentationPresetTest(base_size, mean, std)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
def main(args):
    # 初始化
    palette,pallette_dict = palette_init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path=create_save_path(args.weights_path)
    # 加载数据
    pre_dataset= VOCSegmentation(args.data_path,
                                  year="2007",
                                  transforms=None,
                                  txt_name="test.txt")
    pre_imgs,gt = pre_dataset.images,pre_dataset.masks
    data_transform = transforms.Compose([
        transforms.Resize(args.img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    # 加载模型
    model = torch.load(args.weights_path).to(device)
    model.eval()
    # 开始预测
    time_list = []
    for i,imgpath in enumerate(pre_imgs):
        # 转换tensor图片
        img = Image.open(imgpath)
        # 保存原大小
        ori_w, ori_h = img.size
        img = img.resize((args.img_size, args.img_size))
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            if i == 0:
                # 预热
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)
            #     开始计时
            t_start = time_synchronized()
            out = model(img.to(device))
            t_end = time_synchronized()
            time_list.append(t_end - t_start)
            print("推理时间: {:1.4f}".format(t_end - t_start))
        # 将预测结果转换为图片
        out = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        out_classindex = np.unique(out)[1:]# 去除背景

        mask = Image.fromarray(out)
        mask.putpalette(palette)
        # mask.show()
        mask = mask.convert('RGB')
        # 融合图像使用的cv格式mask
        mask_cv = np.array(mask)[..., ::-1]
        # 将mask图像还原到原图大小
        mask_cv = cv2.resize(mask_cv, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        # 在顶部加入标签信息
        for j in out_classindex:
            label = args.classes[j-1]
            color_rgb = pallette_dict[str(j)]  # RGB 颜色值为红色
            # 将 RGB 颜色值转换为 BGR 颜色值
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.putText(mask_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)


        # print("mask_cv.shape: ", mask_cv.shape)
        # 保存图片
        img_out_path = os.path.join(save_path, os.path.basename(imgpath))
        print("img_out_path: ", img_out_path)
        # 保存图片的方式
        if args.method == "fusion":
            # 图像融合
            ori_img = cv2.imread(imgpath)
            dst = cv2.addWeighted(ori_img, 0.7, mask_cv, 0.3, 0)
            cv2.imwrite(img_out_path, dst)
        elif args.method == "mask":
            # 保存图像分割后的黑白结果图像
            # cv2.imwrite(img_out_path, pr_mask)
            # 保存图像分割后的彩色结果图像
            cv2.imwrite(img_out_path, mask_cv)
        elif args.method == "contours":
            # 找到预测图中缺陷轮廓信息
            pred_img = cv2.cvtColor(mask_cv, cv2.COLOR_RGB2GRAY)
            ori_img = cv2.imread(imgpath)
            result_img = contours_process_nolabel(ori_img, pred_img)
            cv2.imwrite(img_out_path, result_img)
    print("平均时间: {:.3f}ms".format(sum(time_list)/len(time_list)*100))
    #求FPS
    print("FPS: {:.4f}".format(1/(sum(time_list)/len(time_list))))

def parse_args():
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    parser.add_argument("--weights_path", default=r'D:\Files\_Weights\segzoo\fin_paper2\04-24_04-45-52-X_unet_fin_all/best_model.pth', type=str, help="权重路径")
    parser.add_argument("--data_path", default=r'..\VOCdevkit_cap_c5_bin', help="VOCdevkit 路径")
    parser.add_argument("--classes", default=['class1'], help="类别名")
    parser.add_argument("--img-size", default=256, type=int,help="图片缩放大小")
    parser.add_argument("--method", default="fusion",  choices=["fusion", "mask", "contours"], help="输出方式")


    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
