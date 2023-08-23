import os
import random
import time
import datetime
import gc

import numpy as np
import torch
from torch import nn

from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from utils.my_dataset import VOCSegmentation
from utils import transforms as T
import csv
from torch.utils.tensorboard import SummaryWriter
from utils.mytools import calculater_1,Time_calculater
from src.unet_mod import *
from src.nets import *


# 数据集预处理
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),brightness=0.5):
        self.transforms = T.Compose([
            T.Resize(base_size),
            T.ToTensor(),
            T.ColorJitter(brightness=brightness),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(args,train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = args.base_size
    crop_size = args.crop_size

    return SegmentationPresetEval(base_size,mean=mean, std=std,brightness=args.brightness)

def main(args):
    #-----------------------初始化-----------------------

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    #-----------------------加载数据-----------------------
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2007",
                                  transforms=get_transform(args, train=False),
                                  txt_name="val.txt")
    num_workers = 0
    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    #-----------------------创建模型-----------------------
    model = torch.load(args.pretrained)
    # ----------进行预测
    confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
    # -----------------------保存日志-----------------------
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
    val_log=confmat
    val_log["dice loss"]=format(dice, '.4f')
    print('--val_log:',val_log)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--base_size", default=256, type=int, help="图片缩放大小")
    parser.add_argument("--crop_size", default=256,  type=int, help="图片裁剪大小")
    parser.add_argument('--pretrained', default=r'logs/08-20_15-25-37-X_unet_fin_all8/best_model.pth',help='预训练模型路径')
    parser.add_argument("--num-classes", default=6, type=int)

    parser.add_argument("--data-path", default=r"E:\datasets\_using\VOC_MLCC_6_multi", help="VOC数据集路径")
    # parser.add_argument("--data-path", default=r"E:\datasets\_using\VOC_MLCCn6", help="VOC数据集路径")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--brightness", default=0.4, help="number of total epochs to train")


    args = parser.parse_args()

    return args

# tensorboard --logdir logs
# http://localhost:6006/
if __name__ == '__main__':
    args = parse_args()
    main(args)


