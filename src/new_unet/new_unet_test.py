# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
from utils.mytools import model_test
import torch.nn.functional as F
from src.unet_mod.block import Conv,C3



class Down_fine(nn.Sequential):
    def __init__(self, in_channels, out_channels,s=2,n=3):
        mid = out_channels // 2
        super().__init__(
            Conv(in_channels, mid,k=3, s=s),
            C3(mid, out_channels,n=n)
        )
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = C3(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = C3(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
class unet_t1(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = Down_fine(in_channels, base_c, s=1, n=1)
        self.down1 = Down_fine(base_c, base_c * 2, n=1)
        self.down2 = Down_fine(base_c * 2, base_c * 4, n=1)
        self.down3 = Down_fine(base_c * 4, base_c * 8, n=1)
        self.down4 = Down_fine(base_c * 8, base_c * 16, n=1)

        self.up1 = Up(base_c * 16, base_c * 8 )
        self.up2 = Up(base_c * 8, base_c * 4 )
        self.up3 = Up(base_c * 4, base_c * 2 )
        self.up4 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        out = self.out_conv(x)
        return out

if __name__ == '__main__':
    model = unet_t1(3,8)
    model_test(model,(2,3,256,256),'params')
    model_test(model,(2,3,256,256),'shape')