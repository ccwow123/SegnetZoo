# -*- coding: utf-8 -*-
import torch


def export_onnx(model, input, onnx_file_name):
    model.eval()
    torch.onnx.export(model, input, onnx_file_name, verbose=True, opset_version=11)
    print('Onnx export success, saved as %s' % onnx_file_name)


if __name__ == '__main__':
    weights_path = r'logs/05-10_10-26-36-X_unet_fin_all8_CPFFM/best_model.pth'
    out_path = r'onxx/X_unet_fin_all8_CARAFE.onnx'
    # 1. Load model
    model = torch.load(weights_path).to('cuda')
    # 2. Input to the model
    input = torch.randn(1, 3, 256, 256).to('cuda')
    # 3. Export the model
    export_onnx(model, input, out_path)
