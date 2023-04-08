# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from src.unet_mod.block import C3, C3Ghost,C2f,Conv,SPPF, SPP,DWConv,CBAM,SimAM,CoordAtt,C3CSP,AFF
from utils.mytools import model_test
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, 3, 2, 1)
        self.conv2 = C3(out_ch, out_ch, n=3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class Up(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     # nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     # nn.BatchNorm2d(out_ch),
        #     # nn.ReLU(inplace=True)
        # )
        # self.conv = Conv_b(in_ch, out_ch)#原来的
        self.conv1 = Conv(in_ch, out_ch, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = C3(in_ch, in_ch//2, n=3,shortcut=False)

    def forward(self, x1,x2):
        x1 = self.conv1(x1)#512,8,8 -> 256,8,8
        x1 = self.up(x1)    #256,8,8 -> 256,16,16
        x = torch.cat([x2, x1], dim=1)
        x = self.conv2(x)
        # x1 = self.up(x1)#512,128,128 -> 512,256,256
        # # # [N, C, H, W]
        # # diff_y = x2.size()[2] - x1.size()[2]
        # # diff_x = x2.size()[3] - x1.size()[3]
        # #
        # # # padding_left, padding_right, padding_top, padding_bottom
        # # x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
        # #                 diff_y // 2, diff_y - diff_y // 2])
        #
        # x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x)
        return x
# 基础版
class X_Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, base_c=32):
        super().__init__()
        #          0,     1,         2,        3,         4
        #           32,     64,         128,        256,         512
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        self.Conv0 = Conv(in_ch, filters[0], 6, 2, 2)#拓展通道数
        self.Conv1 = Down(filters[0], filters[1])
        self.Conv2 = Down(filters[1], filters[2])
        self.Conv3 = Down(filters[2], filters[3])
        self.Conv = Down(filters[3], filters[4])

        self.spp = SPPF(filters[4], filters[4])

        self.Up3 = Up(filters[4], filters[3])
        self.Up2 = Up(filters[3], filters[2])
        self.Up1 = Up(filters[2], filters[1])
        self.Up0 = Up(filters[1], filters[0])
        self.segmentation_head = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0, bias=True))


    def forward(self, x):
        x0 = self.Conv0(x)#3,256,256 -> 32,128,128
        x1 = self.Conv1(x0)#32,128,128 -> 64,64,64
        x2 = self.Conv2(x1)#64,64,64 -> 128,32,32
        x3 = self.Conv3(x2)#128,32,32 -> 256,16,16
        x_mid = self.Conv(x3)#256,16,16 -> 512,8,8

        x_mid = self.spp(x_mid)#512,8,8 -> 512,8,8

        y_3 = self.Up3(x_mid,x3)#512,8,8 -> 256,16,16
        y_2 = self.Up2(y_3,x2)#256,16,16 -> 128,32,32
        y_1 = self.Up1(y_2,x1)#128,32,32 -> 64,64,64
        y_0 = self.Up0(y_1,x0)#64,64,64 -> 32,128,128
        out = self.segmentation_head(y_0)#32,128,128 -> 2,256,256

        return out


if __name__ == '__main__':
    model = X_Unet()
    model_test(model, (2,3,256,256),'params')