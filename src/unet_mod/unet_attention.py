# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.unet_mod.block import CBAM, SELayer,sa_layer,CoordAtt,SimAM
from src.unet_mod import Unet_EX
from utils.mytools import model_test

class Unet_Attention(Unet_EX):
    def __init__(self,in_ch=3, out_ch=2,
                 base_c: int = 32,
                 block_type='unet', attention=None):
        super().__init__(in_ch, out_ch, base_c, block_type)
        if attention == 'cbam':
            self.att0 = CBAM(channel=base_c)
            self.att1 = CBAM(channel=base_c * 2)
            self.att2 = CBAM(channel=base_c * 4)
            self.att3 = CBAM(channel=base_c * 8)
        elif attention == 'se':
            self.att0 = SELayer(channel=base_c)
            self.att1 = SELayer(channel=base_c * 2)
            self.att2 = SELayer(channel=base_c * 4)
            self.att3 = SELayer(channel=base_c * 8)

        elif attention == 'ca':
            self.att0 = CoordAtt(base_c,base_c)
            self.att1 = CoordAtt(base_c * 2,base_c * 2)
            self.att2 = CoordAtt(base_c * 4,base_c * 4)
            self.att3 = CoordAtt(base_c * 8,base_c * 8)
        elif attention == 'simam':
            self.att0 = SimAM(base_c,base_c)
            self.att1 = SimAM(base_c * 2,base_c * 2)
            self.att2 = SimAM(base_c * 4,base_c * 4)
            self.att3 = SimAM(base_c * 8,base_c * 8)


    def forward(self, x):
        x_1 = self.Conv1(x)  #32，256，256
        x_1 = self.att0(x_1) + x_1
        x_2 = self.Conv2(self.Maxpool(x_1)) #64，128，128
        x_2 = self.att1(x_2) + x_2
        x_3 = self.Conv3(self.Maxpool(x_2))#128，64，64
        x_3 = self.att2(x_3) + x_3
        x_4 = self.Conv4(self.Maxpool(x_3))#256，32，32
        x_4 = self.att3(x_4) + x_4
        x_5 = self.Conv5(self.Maxpool(x_4))#512，16，16
        x_5 = self.spp(x_5)#512，16，16

        y_4 = self.Up5(x_5,x_4)#256，32，32
        y_3 = self.Up4(y_4,x_3)#128，64，64
        y_2 = self.Up3(y_3,x_2)#64，128，128
        y_1 = self.Up2(y_2,x_1)#32，256，256

        out = self.out(y_1)

        return out

if __name__ == "__main__":
    model = Unet_Attention(3,2,attention='simam')
    model_test(model,(2,3,256,256),'shape')