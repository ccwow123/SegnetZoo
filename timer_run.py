from threading import Timer
# from train_dev2 import *
from train import *
import torch
from time import sleep
# 设置延时时间
def set_timer(hour=0, min=0, sec=0):
    # 小时转秒
    def hour2sec(hour):
        return hour * 60 * 60
    # 分钟转秒
    def min2sec(min):
        return min * 60
    return hour2sec(hour) + min2sec(min) + sec
# 执行单个train
def loop(cfg_path):
    torch.cuda.empty_cache()
    args = parse_args(cfg_path)
    main(args)
    torch.cuda.empty_cache()
# 执行多个train
def my_job(jobs,repeat=1):
    for key,_ in jobs.items():
        for i in range(repeat):
            print('-' * 50, '现在执行：', key, '-' * 50)
            loop(key)
            sleep(5)
if __name__ == '__main__':
    repeat = 5 #重复次数
    jobs ={
        "unet": '',
        "Unet0": '',

    }

    Timer(set_timer(sec=1),my_job,(jobs,repeat)).start()

