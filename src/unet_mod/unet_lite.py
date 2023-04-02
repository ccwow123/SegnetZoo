# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from src.unet_mod.block import Bneck,SPPF
from utils.mytools import model_test

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch,conv):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = conv(in_ch, out_ch)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # # [N, C, H, W]
        # diff_y = x2.size()[2] - x1.size()[2]
        # diff_x = x2.size()[3] - x1.size()[3]
        #
        # # padding_left, padding_right, padding_top, padding_bottom
        # x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
        #                 diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Unet_lite(nn.Module):
    def __init__(self, in_ch=3, out_ch=2,
                 base_c: int = 32,
                 block_type='mobile'):
        super().__init__()
        #          32, 64, 128, 256, 512
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        # 编码器
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        if block_type == 'mobile2':
            self.Conv2 = nn.Sequential(Bneck(filters[0], operator_kernel=3,exp_size=filters[0],out_size=filters[0],NL='HS',s=1,SE=True),
                                        Bneck(filters[0], operator_kernel=3,exp_size=filters[0]*4,out_size=filters[1],NL='HS',s=1,SE=True),
                                        Bneck(filters[1], operator_kernel=3,exp_size=filters[0]*8,out_size=filters[1],NL='HS',s=1,SE=True))
            self.Conv3 = nn.Sequential(Bneck(filters[1], operator_kernel=3,exp_size=filters[1],out_size=filters[1],NL='HS',s=1,SE=True),
                                        Bneck(filters[1], operator_kernel=3,exp_size=filters[1]*4,out_size=filters[2],NL='HS',s=1,SE=True),
                                        Bneck(filters[2], operator_kernel=3,exp_size=filters[1]*8,out_size=filters[2],NL='HS',s=1,SE=True))
            self.Conv4 = nn.Sequential(Bneck(filters[2], operator_kernel=3,exp_size=filters[2],out_size=filters[2],NL='HS',s=1,SE=True),
                                        Bneck(filters[2], operator_kernel=3,exp_size=filters[2]*4,out_size=filters[3],NL='HS',s=1,SE=True),
                                        Bneck(filters[3], operator_kernel=3,exp_size=filters[2]*8,out_size=filters[3],NL='HS',s=1,SE=True))
            self.Conv5 = nn.Sequential(Bneck(filters[3], operator_kernel=3,exp_size=filters[3],out_size=filters[3],NL='HS',s=1,SE=True),
                                        Bneck(filters[3], operator_kernel=3,exp_size=filters[3]*4,out_size=filters[4],NL='HS',s=1,SE=True),
                                        Bneck(filters[4], operator_kernel=3,exp_size=filters[3]*8,out_size=filters[4],NL='HS',s=1,SE=True))
        elif block_type == 'mobile1':
            self.Conv2 = Bneck(filters[0], operator_kernel=3,exp_size=filters[0]*4,out_size=filters[1],NL='HS',s=1,SE=True)
            self.Conv3 = Bneck(filters[1], operator_kernel=3,exp_size=filters[1]*4,out_size=filters[2],NL='HS',s=1,SE=True)
            self.Conv4 = Bneck(filters[2], operator_kernel=3,exp_size=filters[2]*4,out_size=filters[3],NL='HS',s=1,SE=True)
            self.Conv5 = Bneck(filters[3], operator_kernel=3,exp_size=filters[3]*4,out_size=filters[4],NL='HS',s=1,SE=True)
        else:
            raise ValueError('block_type must be mobile or mobile2')
        self.spp = SPPF(filters[4], filters[4])
        # 解码器
        self.Up5 = up_conv(filters[4], filters[3],conv_block)
        self.Up4 = up_conv(filters[3], filters[2],conv_block)
        self.Up3 = up_conv(filters[2], filters[1],conv_block)
        self.Up2 = up_conv(filters[1], filters[0],conv_block)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 编码器
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # 解码器
        x = self.Up5(x5, x4)
        x = self.Up4(x, x3)
        x = self.Up3(x, x2)
        x = self.Up2(x, x1)
        out = self.Conv(x)
        return out


if __name__ == "__main__":
    model = Unet_lite(3, 2, block_type='mobile2')
    model_test(model, (2, 3, 256, 256), 'params')