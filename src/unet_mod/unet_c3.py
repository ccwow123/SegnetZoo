# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from src.unet_mod.block import C3, C3Ghost,C2f,Conv,SPPF, SPP,DWConv
from utils.mytools import model_test

#----------------#
#     yolo_Unet
#----------------#

class up_C3(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch,Conv_b):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = Conv_b(in_ch, out_ch)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Unet_c3g(nn.Module):
    def __init__(self, in_ch=3, out_ch=2,base_c: int = 32,block: str = 'C3Ghost'):
        super().__init__()
        #           64, 128, 256, 512, 1024
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        if block == 'C3Ghost':
            Conv_b = C3Ghost
        elif block == 'C3':
            Conv_b = C3
        elif block == 'C2f':
            Conv_b = C2f
        else:
            raise NotImplementedError(f'block {block} is not implemented')
        # 编码器
        self.Conv1 =Conv(in_ch, filters[0], 6, 2, 2)
        self.Conv2 =nn.Sequential(Conv(filters[0], filters[1], 3, 2, 1),
                                  Conv_b(filters[1], filters[1]))
        self.Conv3 =nn.Sequential(Conv(filters[1], filters[2], 3, 2, 1),
                                    Conv_b(filters[2], filters[2]))
        self.Conv4 =nn.Sequential(Conv(filters[2], filters[3], 3, 2, 1),
                                    Conv_b(filters[3], filters[3]))
        self.Conv5 =nn.Sequential(Conv(filters[3], filters[4], 3, 2, 1),
                                    Conv_b(filters[4], filters[4]))
        self.SPP = SPPF(filters[4], filters[4])
        # 解码器
        self.Up4 = up_C3(filters[4], filters[3],Conv_b)
        self.Up3 = up_C3(filters[3], filters[2],Conv_b)
        self.Up2 = up_C3(filters[2], filters[1],Conv_b)
        self.Up1 = up_C3(filters[1], filters[0],Conv_b)
        self.Up0 = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(filters[0]),
                        nn.ReLU(inplace=True)
                    )
        self.segmentation_head = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_1 = self.Conv1(x)#128, 160, 160
        x_2 = self.Conv2(x_1)#256, 80, 80
        x_3 = self.Conv3(x_2)#512, 40, 40
        x_4 = self.Conv4(x_3)#1024, 20, 20
        # mid
        x_5 = self.Conv5(x_4)#1024, 20, 20
        x_mid = self.SPP(x_5)#1024, 20, 20

        y_4 = self.Up4(x_mid,x_4)#512, 40, 40
        y_3 = self.Up3(y_4,x_3)#256, 80, 80
        y_2 = self.Up2(y_3,x_2)#128, 160, 160
        y_1 = self.Up1(y_2,x_1)#64, 320, 320
        y_0 = self.Up0(y_1)#32, 640, 640

        out = self.segmentation_head(y_0)
        return out

class Unet_best(nn.Module):
    def __init__(self, in_ch=3, out_ch=2,base_c: int = 32,block: str = 'C3',spp='sppf',dw=False):
        super().__init__()
        #           64, 128, 256, 512, 1024
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]

        if dw :
            Conv = DWConv

        if block == 'C3Ghost':
            Conv_b = C3Ghost
        elif block == 'C3':
            Conv_b = C3
        else:
            raise NotImplementedError(f'block {block} is not implemented')
        # 编码器
        self.Conv1 =Conv(in_ch, filters[0], 6, 2, 2)
        self.Conv2 =nn.Sequential(Conv(filters[0], filters[1], 3, 2, 1),
                                  Conv_b(filters[1], filters[1]))
        self.Conv3 =nn.Sequential(Conv(filters[1], filters[2], 3, 2, 1),
                                    Conv_b(filters[2], filters[2]))
        self.Conv4 =nn.Sequential(Conv(filters[2], filters[3], 3, 2, 1),
                                    Conv_b(filters[3], filters[3]))
        self.Conv5 =nn.Sequential(Conv(filters[3], filters[4], 3, 2, 1),
                                    Conv_b(filters[4], filters[4]))

        if spp == 'sppf':
            self.SPP = SPPF(filters[4], filters[4])
        elif spp == 'spp':
            self.SPP = SPP(filters[4], filters[4])
        else:
            raise NotImplementedError(f'spp {spp} is not implemented')
        # 解码器
        self.Up4 = up_C3(filters[4], filters[3],Conv_b)
        self.Up3 = up_C3(filters[3], filters[2],Conv_b)
        self.Up2 = up_C3(filters[2], filters[1],Conv_b)
        self.Up1 = up_C3(filters[1], filters[0],Conv_b)
        self.Up0 = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(filters[0]),
                        nn.ReLU(inplace=True)
                    )
        self.segmentation_head = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_1 = self.Conv1(x)#128, 160, 160
        x_2 = self.Conv2(x_1)#256, 80, 80
        x_3 = self.Conv3(x_2)#512, 40, 40
        x_4 = self.Conv4(x_3)#1024, 20, 20
        # mid
        x_5 = self.Conv5(x_4)#1024, 20, 20
        x_mid = self.SPP(x_5)#1024, 20, 20

        y_4 = self.Up4(x_mid,x_4)#512, 40, 40
        y_3 = self.Up3(y_4,x_3)#256, 80, 80
        y_2 = self.Up2(y_3,x_2)#128, 160, 160
        y_1 = self.Up1(y_2,x_1)#64, 320, 320
        y_0 = self.Up0(y_1)#32, 640, 640

        out = self.segmentation_head(y_0)
        return out
if __name__ == "__main__":
    # model = Unet_c3g(3,2,block='C3')
    model = Unet_best(3,2,block='C3',spp='spp',dw=True)
    model_test(model,(2,3,256,256),'params')