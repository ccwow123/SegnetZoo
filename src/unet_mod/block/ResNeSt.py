import torch
import torch.nn as nn
import torch.nn.functional as F


class SplAtConv2d(nn.Module):
    """
    Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SplAtConv2d, self).__init__()
        # #
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.radix_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                      groups=groups * radix, bias=bias, **kwargs),
            norm_layer(channels * radix),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.relu = nn.ReLU(inplace=True)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        # -------------------------------
        # 经过radix_conv即组卷积产生multi branch个分支U
        # U等分成radix个组，组求和得到gap通道内的值
        x = self.radix_conv(x)
        batch, rchannel = x.shape[:2]
        splited = torch.split(x, rchannel // self.radix, dim=1)
        gap = sum(splited)
        # -------------------------------
        # gap通道内 avgpool + fc1 + fc2 + softmax
        # 其中softmax是对radix维度进行softmax
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        # -------------------------------
        # 将gap通道计算出的和注意力和原始分出的radix组个branchs相加得到最后结果
        attens = torch.split(atten, rchannel // self.radix, dim=1)
        out = sum([att * split for (att, split) in zip(attens, splited)])
        # -------------------------------
        # 返回一个out的copy, 使用contiguous是保证存储顺序的问题
        return out.contiguous()


# 对radix维度进行softmax
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)

        x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        # x: [Batchsize, radix, cardinality, h, w]
        x = F.softmax(x, dim=1)  # 对radix维度进行softmax
        x = x.reshape(batch, -1)

        return x


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    # expansion = 4
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        # 组卷积中组的个数 = 输出channel * cardinality的个数
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        # 1x1 组卷积
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        # 用来判断是否是block的连接处
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        # split attention conv
        self.conv2 = SplAtConv2d(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=dilation, groups=cardinality, bias=False,
            radix=radix, norm_layer=norm_layer)
        # 1x1 组卷积
        self.conv3 = nn.Conv2d(
            group_width, planes , kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample==None:
            self.downsample=nn.Sequential(
                            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
                            nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
                            nn.BatchNorm2d(planes , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        # ------------------------------------
        # 用1x1 组卷积等效 multi branchs
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        # ------------------------------------
        # Split attention
        out = self.conv2(out)
        # ------------------------------------
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        # ------------------------------------
        # 1X1 conv + bn
        out = self.conv3(out)
        out = self.bn3(out)
        # ------------------------------------
        # 跟resnet一样，block和block之间的第一个需要residual需要downsample来降维
        # 这里downsample方法为resnet-D中AvgPool(2) + 1x1卷积
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # ------------------------------------
        return out


if __name__ == '__main__':
    downsample = nn.Sequential(
        nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
        nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    model = Bottleneck(64,256)
    # model = Bottleneck(64,64,downsample=downsample)
    x = torch.randn(2, 64, 512, 512)
    y = model(x)
    print(y.shape)
# -*- coding: utf-8 -*-
