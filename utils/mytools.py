# -*- coding: utf-8 -*-
import datetime
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from thop import profile
from torchsummary import summary
# import wandb
class Time_calculater(object):
    def __init__(self):
        self.start=time.time()
        self.last_time=self.start
        self.remain_time=0
    #定义将秒转换为时分秒格式的函数
    def time_change(self,time_init):
        time_list = []
        if time_init/3600 > 1:
            time_h = int(time_init/3600)
            time_m = int((time_init-time_h*3600) / 60)
            time_s = int(time_init - time_h * 3600 - time_m * 60)
            time_list.append(str(time_h))
            time_list.append('h ')
            time_list.append(str(time_m))
            time_list.append('m ')

        elif time_init/60 > 1:
            time_m = int(time_init/60)
            time_s = int(time_init - time_m * 60)
            time_list.append(str(time_m))
            time_list.append('m ')
        else:
            time_s = int(time_init)

        time_list.append(str(time_s))
        time_list.append('s')
        time_str = ''.join(time_list)
        return time_str
    def time_cal(self,i,N):
        now_time=time.time()
        self.remain_time=(now_time-self.last_time)*(N-i-1)
        self.last_time=now_time
        print('%'*20,"剩余时间："+self.time_change(self.remain_time),'%'*20)

def calculater_1(model, input_size=(3, 512, 512), device='cuda'):
    # model = torchvision.models.alexnet(pretrained=False)
    # dummy_input = torch.randn(1, 3, 224, 224)
    # dummy_input = torch.randn(1, *input_size).cuda()
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, (dummy_input,))
    flops, params = (flops / 1e9), (params / 1e6)
    print('flops: %.2f G' % flops)
    print('params: %.2f M' % params)
    return flops, params
def model_test(model,input_size,method, device='cuda'):
    '''

    Args:
        model: 要进行测试的模型
        input_size:  输入tensor的尺寸 (2,3,256,256)
        method:  测试方法，shape/summary/params
        device:  测试设备，cuda/cpu

    Returns: None

    '''
    from thop import profile
    from torchsummary import summary
    def calculater_1(model, input_size=(3, 512, 512), device='cuda'):
        # model = torchvision.models.alexnet(pretrained=False)
        # dummy_input = torch.randn(1, 3, 224, 224)
        # dummy_input = torch.randn(1, *input_size).cuda()
        dummy_input = torch.randn(1, *input_size).to(device)
        flops, params = profile(model, (dummy_input,))
        print('flops: %.2fG' % (flops / 1e9))
        print('params: %.2fM' % (params / 1e6))
        return flops / 1e9, params / 1e6

    model = model.to(device)
    if method == 'shape':
        input = torch.randn(input_size).to(device)
        out = model(input)
        print('out.shape:', out.shape)
    elif method == 'summary':
        summary(model, input_size[1:])
    elif method == 'params':
        calculater_1(model, input_size[1:], device=device)


if __name__ == '__main__':
    time_calculater=Time_calculater()
    N=10#实际使用时用相应变量替换掉
    for i in range(N):
        time.sleep(1)#为了测试效果添加的
        time_calculater.time_cal(i,N)