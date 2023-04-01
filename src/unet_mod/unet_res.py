# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from src.unet_mod.block import SPPF ,Bottleneck
from utils.mytools import model_test

#----------------#
# resnet_block
#----------------#
class BasicBlock(nn.Module):
    expansion = 1
    '''
    expansion通道扩充比例
    out_channels就是输出的channel
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

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

class Unet_EX(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=2,
                 base_c: int = 32,
                 block_type='unet'):
        super().__init__()
        #          32, 64, 128, 256, 512
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        # 编码器
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        if block_type=='unet':
            self.Conv2 = conv_block(filters[0], filters[1])
            self.Conv3 = conv_block(filters[1], filters[2])
            self.Conv4 = conv_block(filters[2], filters[3])
            self.Conv5 = conv_block(filters[3], filters[4])
        elif block_type=='resnet':
            self.Conv2 = BasicBlock(filters[0], filters[1])
            self.Conv3 = BasicBlock(filters[1], filters[2])
            self.Conv4 = BasicBlock(filters[2], filters[3])
            self.Conv5 = BasicBlock(filters[3], filters[4])
        elif block_type == 'resnest':
            self.Conv2 = Bottleneck(filters[0], filters[1])
            self.Conv3 = Bottleneck(filters[1], filters[2])
            self.Conv4 = Bottleneck(filters[2], filters[3])
            self.Conv5 = Bottleneck(filters[3], filters[4])
        else:
            raise NotImplementedError('block_type 不存在')
        self.spp = SPPF(filters[4], filters[4])
        # 解码器
        self.Up5 = up_conv(filters[4], filters[3],conv_block)
        self.Up4 = up_conv(filters[3], filters[2],conv_block)
        self.Up3 = up_conv(filters[2], filters[1],conv_block)
        self.Up2 = up_conv(filters[1], filters[0],conv_block)
        self.out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_1 = self.Conv1(x)  #32，256，256
        x_2 = self.Conv2(self.Maxpool(x_1)) #64，128，128
        x_3 = self.Conv3(self.Maxpool(x_2))#128，64，64
        x_4 = self.Conv4(self.Maxpool(x_3))#256，32，32
        x_5 = self.Conv5(self.Maxpool(x_4))#512，16，16
        x_5 = self.spp(x_5)#512，16，16

        y_4 = self.Up5(x_5,x_4)#256，32，32
        y_3 = self.Up4(y_4,x_3)#128，64，64
        y_2 = self.Up3(y_3,x_2)#64，128，128
        y_1 = self.Up2(y_2,x_1)#32，256，256

        out = self.out(y_1)
        return out

if __name__ == "__main__":
    model = Unet_EX(3,2,block_type='resnest')
    model_test(model,(2,3,256,256),'shape')