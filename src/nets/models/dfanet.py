""" Deep Feature Aggregation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import Enc, FCAttention, get_xception_a
from .base_models import resnet101

__all__ = ['DFANet', 'get_dfanet', 'get_dfanet_citys']

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(False) if relu6 else nn.ReLU(False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DFANet(nn.Module):
    def __init__(self, nclass, backbone='xception', aux=False, jpu=False, pretrained_base=False, **kwargs):
        super(DFANet, self).__init__()
        self.pretrained = get_xception_a(pretrained_base, **kwargs)


        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)

        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)

        self.enc2_1_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_2_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_3_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.conv_fusion = _ConvBNReLU(32, 32, 1, **kwargs)

        self.fca_1_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_2_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_3_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, nclass, 1)

        self.__setattr__('exclusive', ['enc2_2', 'enc3_2', 'enc4_2', 'fca_2', 'enc2_3', 'enc3_3', 'enc3_4', 'fca_3',
                                       'enc2_1_reduce', 'enc2_2_reduce', 'enc2_3_reduce', 'conv_fusion', 'fca_1_reduce',
                                       'fca_2_reduce', 'fca_3_reduce', 'conv_out'])

    def forward(self, x):
        size = x.size()[2:]
        # backbone
        stage1_conv1 = self.pretrained.conv1(x)
        stage1_enc2 = self.pretrained.enc2(stage1_conv1)
        stage1_enc3 = self.pretrained.enc3(stage1_enc2)
        stage1_enc4 = self.pretrained.enc4(stage1_enc3)
        stage1_fca = self.pretrained.fca(stage1_enc4)
        stage1_out = F.interpolate(stage1_fca, stage1_enc2.size()[2:], mode='bilinear', align_corners=True)
        # F.interpolate(auxout, size, mode='bilinear', align_corners=True)

        # stage2
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(stage2_fca, stage2_enc2.size()[2:], mode='bilinear', align_corners=True)

        # stage3
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)

        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2), stage1_enc2_decoder.size()[2:],
                                            mode='bilinear', align_corners=True)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2), stage1_enc2_decoder.size()[2:],
                                            mode='bilinear', align_corners=True)
        fusion = stage1_enc2_decoder + stage2_enc2_docoder + stage3_enc2_decoder
        fusion = self.conv_fusion(fusion)
        # print(fusion.shape)
        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca),  fusion.size()[2:],
                                           mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca), fusion.size()[2:],
                                           mode='bilinear', align_corners=True)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca), fusion.size()[2:],
                                           mode='bilinear', align_corners=True)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder + stage3_fca_decoder

        outputs = list()
        out = self.conv_out(fusion)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        outputs.append(out)

        # return outputs
        return out

def get_dfanet(num_class=1, backbone='', root='~/.torch/models',
               pretrained_base=True, **kwargs):

    model = DFANet(num_class, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model


def get_dfanet_citys(**kwargs):
    return get_dfanet('citys', **kwargs)


if __name__ == '__main__':
    model = get_dfanet_citys()
