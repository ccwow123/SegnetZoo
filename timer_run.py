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
    args = parse_args(cfg_path)
    main(args)
# 执行多个train
def my_job(jobs,repeat=1):
    for key,_ in jobs.items():
        for i in range(repeat):
            print('-' * 50, '现在执行：', key, '-' * 50)
            loop(key)
            sleep(20)
if __name__ == '__main__':
    repeat = 2 #重复次数
    jobs ={
        "X_unet_fin_all8": '',
        # "Unet0": '',
        # "Unet_mobile_s": '',
        # "Unet_res": '',
        # 'lraspp_mobilenetv3_large': '',
        # "FCN": '',
        # "SegNet": '',
        # "DenseASPP": '',
        # 'deeplabV3p': '',
        #
        # 'X_unet_fin_all8_noall': '',
        # 'X_unet_fin_all8_DA': '',
        # 'X_unet_fin_all8_CARAFE': '',
        # 'X_unet_fin_all8_CPFFM': '',
    }

    Timer(set_timer(sec=1),my_job,(jobs,repeat)).start()
    # Timer(set_timer(hour=5),my_job,(jobs2,repeat)).start()

# "Unet_c3g": '',
# "Unet_c3": '',
# "Unet_c2f": '',
# "Unet_res": '',
# "Unet_resnest": '',
# "Unet_res_cbam": '',
# "Unet_res_se": '',
# "Unet_res_ca": '',
# "Unet_res_simam": '',
# "Unet_mobile_s": '',
# "Unet_shuffle": '',
# 'Unet0_drop': '',#drop=0.3
# 'deeplabV3p': '',
# 'lraspp_mobilenetv3_large': '',
# 'Unet_C3': '',
# 'Unet_C3_spp': '',
# 'Unet_C3_sppf': '',
# 'Unet_C3_sam': '',
# 'Unet_C3_cbam': '',
# 'Unet_C3_sppf_cbam': '',
# 'Unet_C3_sppf_sam': '',
#  'Unet_C3_sppf_ca': '',
# 'Unet_C3_sppf_cbam_r': '',
# 'Unet_C3_sppf_sam_r': '',
# 'Unet_C3_sppf_ca_r': '',
# 'Unet_C3CSP': '',
# 'Unet_C3CSP_sppf': '',
# 'Unet_C3CSP_sppf_sam': '',
# "Unet_C33": '',
# "Unet_C33_sppf": '',
# "Unet_C33_sppf_sam": '',
#
# "Unet_cat": '',
# "Unet_cat_sppf": '',

# 'Unet_C33e': '',
# 'Unet_C33e_sppf': '',
# 'Unet_C33e_sppf_sam': '',