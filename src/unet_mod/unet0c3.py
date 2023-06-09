from typing import Dict
import torch
import torch.nn as nn
from utils.mytools import model_test
import torch.nn.functional as F
from src.unet_mod.block import C3,Conv,SPPF,SimAM,CoordAtt,ASPP,CBAM,BasicRFB

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            C3(in_channels, out_channels,n=3)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
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


class Unet0c3(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits




# v2 不使用pooling
class Down_v2(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            Conv(in_channels, out_channels, 3, 2, 1),
            C3(out_channels, out_channels,n=3)
        )

class Unet0c3_v2(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

# v2_1加入SPPF
class Unet0c3_v2_1(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.sppf = SPPF(base_c * 16, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.sppf(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
# v2_2加入注意力
class Unet0c3_v2_2(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32,
                 attention='ca'):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.sppf = SPPF(base_c * 16, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        if attention == 'ca':
            self.att_1 = CoordAtt(base_c * 1)
            self.att_2 = CoordAtt(base_c * 2)
            self.att_3 = CoordAtt(base_c * 4)
            self.att_4 = CoordAtt(base_c * 8)

    def forward(self, x: torch.Tensor) :
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.sppf(x5)
        x = self.up1(x5, self.att_4(x4))
        x = self.up2(x, self.att_3(x3))
        x = self.up3(x, self.att_2(x2))
        x = self.up4(x, self.att_1(x1))
        logits = self.out_conv(x)
        return logits
# v2_3不使用SPPF，加入注意力
class Unet0c3_v2_3(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32,
                 attention='ca'):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        if attention == 'ca':
            self.att_1 = CoordAtt(base_c * 1)
            self.att_2 = CoordAtt(base_c * 2)
            self.att_3 = CoordAtt(base_c * 4)
            self.att_4 = CoordAtt(base_c * 8)
        elif attention == 'CBAM':
            self.att_1 = CBAM(base_c * 1)
            self.att_2 = CBAM(base_c * 2)
            self.att_3 = CBAM(base_c * 4)
            self.att_4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) :
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.att_4(x4))
        x = self.up2(x, self.att_3(x3))
        x = self.up3(x, self.att_2(x2))
        x = self.up4(x, self.att_1(x1))
        logits = self.out_conv(x)
        return logits
# v2_4不使用SPPF，加入注意力,注意力加在concat之后
class Unet0c3_v2_4(nn.Module):
        def __init__(self,
                     in_channels: int = 3,
                     num_classes: int = 2,
                     bilinear: bool = False,
                     base_c: int = 32,
                     attention='ca'):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.bilinear = bilinear

            self.in_conv = Conv(in_channels, base_c)
            self.down1 = Down(base_c, base_c * 2)
            self.down2 = Down(base_c * 2, base_c * 4)
            self.down3 = Down(base_c * 4, base_c * 8)
            factor = 2 if bilinear else 1
            self.down4 = Down(base_c * 8, base_c * 16 // factor)
            self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
            self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
            self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
            self.up4 = Up(base_c * 2, base_c, bilinear)
            self.out_conv = OutConv(base_c, num_classes)

            if attention == 'ca':
                self.att_1 = CoordAtt(base_c * 1)
                self.att_2 = CoordAtt(base_c * 2)
                self.att_3 = CoordAtt(base_c * 4)
                self.att_4 = CoordAtt(base_c * 8)
            elif attention == 'CBAM':
                self.att_1 = CBAM(base_c * 1)
                self.att_2 = CBAM(base_c * 2)
                self.att_3 = CBAM(base_c * 4)
                self.att_4 = CBAM(base_c * 8)


        def forward(self, x: torch.Tensor):
            x1 = self.in_conv(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.att_4(self.up1(x5, x4))
            x = self.att_3(self.up2(x, x3))
            x = self.att_2(self.up3(x, x2))
            x = self.att_1(self.up4(x, x1))
            logits = self.out_conv(x)
            return logits


#v3 在将中间改为SPPF
class Down_mid(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            C3(in_channels, out_channels*2,n=1),
            SPPF(out_channels*2, out_channels),
            C3(out_channels, out_channels,n=1),
        )
class Unet0c3_v3(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down_mid(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
#v3_1 在将中间改为RFB
class Down_mid_RFB(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            C3(in_channels, out_channels*2,n=1),
            BasicRFB(out_channels*2, out_channels),
            C3(out_channels, out_channels,n=1),
        )
class Unet0c3_v3_1(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down_mid_RFB(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

# v4 使用C3 n=2
class Up_v4(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = C3(in_channels, out_channels,n=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = C3(in_channels, out_channels,n=2)

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

class Down_v4(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            Conv(in_channels, out_channels, 3, 2, 1),
            C3(out_channels, out_channels,n=2)
        )

class Unet0c3_v4(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down_v4(base_c, base_c * 2)
        self.down2 = Down_v4(base_c * 2, base_c * 4)
        self.down3 = Down_v4(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down_v4(base_c * 8, base_c * 16 // factor)
        self.up1 = Up_v4(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up_v4(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up_v4(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up_v4(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

# v5 使用sppf，bilinear=True,attention='cbam'
class Unet0c3_v5(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32,
                 attention='ca'):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        if attention == 'ca':
            self.att_1 = CoordAtt(base_c * 1)
            self.att_2 = CoordAtt(base_c * 2)
            self.att_3 = CoordAtt(base_c * 4)
            self.att_4 = CoordAtt(base_c * 8)
        elif attention == 'CBAM':
            self.att_1 = CBAM(base_c * 1)
            self.att_2 = CBAM(base_c * 1)
            self.att_3 = CBAM(base_c * 2)
            self.att_4 = CBAM(base_c * 4)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.att_4(x)
        x = self.up2(x, x3)
        x = self.att_3(x)
        x = self.up3(x, x2)
        x = self.att_2(x)
        x = self.up4(x, x1)
        x = self.att_1(x)
        logits = self.out_conv(x)
        return logits

# v6 使用sppf，bilinear=True,attention='cbam'嵌入up中
class Up_v6(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = C3(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = C3(in_channels, out_channels)
        self.att = CBAM(in_channels)

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
        x = self.att(x)
        x = self.conv(x)
        return x
class Unet0c3_v6(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = Conv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up_v6(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up_v6(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up_v6(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up_v6(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
if __name__ == '__main__':
    # model = Unet0c3_v2_4(3,2,attention='S2')
    # model = Unet0c3_v3_1(3,2)
    model = Unet0c3_v6(3,2)
    model_test(model,(2,3,256,256),'shape')