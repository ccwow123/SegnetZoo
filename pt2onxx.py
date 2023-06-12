# -*- coding: utf-8 -*-
import torch
from torchvision import models
import onnx
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

def pt2onnx(pt_path, onnx_path):
    model = torch.load(pt_path)
    model = model.eval().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        torch.onnx.export(
            model,                   # 要转换的模型
            x,                       # 模型的任意一组输入
            onnx_path, # 导出的 ONNX 文件名
            opset_version=11,        # ONNX 算子集版本
            input_names=['input'],   # 输入 Tensor 的名称（自己起名字）
            output_names=['output']  # 输出 Tensor 的名称（自己起名字）
        )
    # 读取 ONNX 模型
    onnx_model = onnx.load(onnx_path)

    # 检查模型格式是否正确
    onnx.checker.check_model(onnx_model)

    print('无报错，onnx模型载入成功')

if __name__ == '__main__':
    pt_path = r'D:\Files\_Weights\segzoo\all\05-12_04-43-21-X_unet_fin_all8_CPFFM\best_model.pth'
    onnx_path = 'onxx_file/X_unet_fin_all8_CPFFM.onnx'
    pt2onnx(pt_path, onnx_path)
    # Netron 可视化 ONNX 模型
    # https://netron.app/