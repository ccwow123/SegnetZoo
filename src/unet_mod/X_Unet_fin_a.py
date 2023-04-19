# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
from utils.mytools import model_test
import torch.nn.functional as F
from src.unet_mod.block import C3,Conv,SPPF,SimAM,CoordAtt,ASPP,CBAM,BasicRFB,SCA
class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor
class SPPCSPC_group(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC_group, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, g=4)
        self.cv2 = Conv(c1, c_, 1, 1, g=4)
        self.cv3 = Conv(c_, c_, 3, 1, g=4)
        self.cv4 = Conv(c_, c_, 1, 1, g=4)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1, g=4)
        self.cv6 = Conv(c_, c_, 3, 1, g=4)
        self.cv7 = Conv(2 * c_, c2, 1, 1, g=4)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class Down_fine(nn.Sequential):
    def __init__(self, in_channels, out_channels,s=2):
        mid = out_channels // 2
        super().__init__(
            Conv(in_channels, mid,k=3, s=s),
            C3(mid, out_channels,n=3)
        )
class Up_fin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = CARAFE(in_channels, in_channels // 2)
        self.conv = C3(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
# 注意力在左encoder
class X_unet_fin_al(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = Down_fine(in_channels, base_c,s=1)
        self.down1 = Down_fine(base_c, base_c * 2)
        self.down2 = Down_fine(base_c * 2, base_c * 4)
        self.down3 = Down_fine(base_c * 4, base_c * 8)

        self.down4 = Down_fine(base_c * 8, base_c * 16 )
        self.middle = SPPF(base_c * 16,  base_c * 16)

        self.up1 = Up_fin(base_c * 16, base_c * 8 )
        self.up2 = Up_fin(base_c * 8, base_c * 4 )
        self.up3 = Up_fin(base_c * 4, base_c * 2 )
        self.up4 = Up_fin(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

        self.att1 = SCA(base_c)
        self.att2 = SCA(base_c * 2)
        self.att3 = SCA(base_c * 4)
        self.att4 = SCA(base_c * 8)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(self.att1(x0))
        x2 = self.down2(self.att2(x1))
        x3 = self.down3(self.att3(x2))
        x4 = self.down4(self.att4(x3))
        x5 = self.middle(x4)
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        out = self.out_conv(x)
        return out
# 注意力在右decoder
class X_unet_fin_ar(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = Down_fine(in_channels, base_c,s=1)
        self.down1 = Down_fine(base_c, base_c * 2)
        self.down2 = Down_fine(base_c * 2, base_c * 4)
        self.down3 = Down_fine(base_c * 4, base_c * 8)

        self.down4 = Down_fine(base_c * 8, base_c * 16 )
        self.middle = SPPF(base_c * 16,  base_c * 16)

        self.up1 = Up_fin(base_c * 16, base_c * 8 )
        self.up2 = Up_fin(base_c * 8, base_c * 4 )
        self.up3 = Up_fin(base_c * 4, base_c * 2 )
        self.up4 = Up_fin(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

        self.att1 = SCA(base_c)
        self.att2 = SCA(base_c * 2)
        self.att3 = SCA(base_c * 4)
        self.att4 = SCA(base_c * 8)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.middle(x4)
        x = self.att4(self.up1(x5, x3))
        x = self.att3(self.up2(x, x2))
        x = self.att2(self.up3(x, x1))
        x = self.att1(self.up4(x, x0))
        out = self.out_conv(x)
        return out

# 注意力在中间
class X_unet_fin_am(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = Down_fine(in_channels, base_c,s=1)
        self.down1 = Down_fine(base_c, base_c * 2)
        self.down2 = Down_fine(base_c * 2, base_c * 4)
        self.down3 = Down_fine(base_c * 4, base_c * 8)

        self.down4 = Down_fine(base_c * 8, base_c * 16 )
        self.middle = SPPF(base_c * 16,  base_c * 16)

        self.up1 = Up_fin(base_c * 16, base_c * 8 )
        self.up2 = Up_fin(base_c * 8, base_c * 4 )
        self.up3 = Up_fin(base_c * 4, base_c * 2 )
        self.up4 = Up_fin(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

        self.att1 = SCA(base_c)
        self.att2 = SCA(base_c * 2)
        self.att3 = SCA(base_c * 4)
        self.att4 = SCA(base_c * 8)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.middle(x4)
        x = self.up1(x5, self.att4(x3))
        x = self.up2(x, self.att3(x2))
        x = self.up3(x, self.att2(x1))
        x = self.up4(x, self.att1(x0))
        out = self.out_conv(x)
        return out
if __name__ == '__main__':
    model = X_unet_fin_ar(3,2)
    model_test(model,(2,3,256,256),'params')
    model_test(model,(2,3,256,256),'shape')





