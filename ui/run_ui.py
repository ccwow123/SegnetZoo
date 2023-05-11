
import json
import os
import sys
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from torchvision import transforms

# 主窗口
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 一些全局变量
        self.ROOT_path = os.getcwd()
        self.output_folder = os.path.join(self.ROOT_path, 'output')
        self.device = 'cpu'
        self.classes = ['E exposure','E skew','P extend','P broken','P Indentation','E sticky impurities']
        self.method = 'mask'
        self.model = None
        # 初始化ui
        self.init_ui()
    def init_ui(self):
        self.ui = uic.loadUi("./myui.ui")
        # print(self.ui.__dict__)  # 查看ui文件中有哪些控件

        # 设置参考尺寸和标题
        self.ui.resize(800, 600)
        # self.ui.setWindowIcon(QIcon(os.getcwd() + '\\data\\source_image\\Detective.ico'))
        self.ui.setWindowTitle("检测平台")

        self.btn_init_trigger = False # 模型初始化按钮触发状态
        self.btn_open_trigger = False # 打开图片按钮触发状态
        self.btn_detectPic_trigger = False # 检测图片按钮触发状态
        # 自动执行的函数
        self.enable_controls()# 启用控件
        self.method_init()# 初始化方法
        self.create_output_folder()# 创建输出文件夹
        self.auto_find_model()# 自动查找模型
        # 绑定信号与槽函数
        self.ui.btn_device.clicked.connect(self.change_device)
        self.ui.btn_init.clicked.connect(self.start_thread_model_init)
        self.ui.btn_open.clicked.connect(self.openimage)
        self.ui.btn_detectPic.clicked.connect(self.detect_image)
        # comboBox刷新
        self.ui.comboBox_method.currentIndexChanged.connect(self.change_method)


    # ++++++++++++++++++++++++主要槽函数++++++++++++++++++++
    def append_text(self, text):
        # 获取textBrowser的滚动条
        scrollbar = self.ui.textBrowser.verticalScrollBar()
        # 将滚动条设置到最底部
        scrollbar.setValue(scrollbar.maximum())
        # show text
        self.ui.textBrowser.append(text)
        # 获取textBrowser的滚动条
        scrollbar = self.ui.textBrowser.verticalScrollBar()
        # 将滚动条设置到最底部
        scrollbar.setValue(scrollbar.maximum())


    def enable_controls(self):
        '''启用控件'''
        self.ui.btn_open.setEnabled(self.btn_init_trigger)
        self.ui.btn_detectPic.setEnabled(self.btn_open_trigger)

    def method_init(self):
        '''输出模式初始化'''
        self.ui.comboBox_method.addItem('mask')
        self.ui.comboBox_method.addItem('fusion')

    def change_method(self):
        '''改变输出模式'''
        self.method = self.ui.comboBox_method.currentText()
        self.ui.textBrowser.append('当前输出模式：' + self.method)

    def create_output_folder(self):
        '''创建输出文件夹'''
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def change_device(self):
        '''切换设备'''
        if self.device == 'cpu':
            self.device = 'cuda'
            self.ui.btn_device.setText('GPU')
        else:
            self.device = 'cpu'
            self.ui.btn_device.setText('CPU')
        self.append_text('当前设备：' + self.device)

    def auto_find_model(self):
        '''自动查找模型'''
        pth_list = os.listdir(os.path.join(self.ROOT_path, 'model'))
        for pth in pth_list:
            if pth.endswith('.pth'):
                self.ui.comboBox_pt.addItem(pth)
        self.append_text('已找到模型：' + str(pth_list))

    def start_thread_model_init(self):
        threading.Thread(target=self.model_init).start()
    def change_model(self):
        '''改变权重'''
        self.model_path = self.ui.comboBox_pt.currentText()
        self.append_text('当前模型：' + self.model_path)
    def model_init(self):
        '''模型初始化'''
        self.btn_init_trigger = True # 模型初始化按钮触发状态为True，代表已经按下
        self.change_model()
        self.append_text('模型初始化中...')
        # 加载模型
        model = torch.load(self.ROOT_path + '/model/' + self.model_path).to(self.device)
        model.eval()
        self.append_text('模型初始化完成！')
        self.model = model
        # 刷新控件状态
        self.enable_controls()

    def openimage(self):
        '''打开图片'''
        def show_image(img_path, win):
            pixmap = QtGui.QPixmap(img_path)
            size = win.size()
            scaled_pixmap = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            win.setPixmap(scaled_pixmap)
        self.btn_open_trigger = True # 打开图片按钮触发状态为True，代表已经按下
        self.img_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.append_text('图片路径：' + self.img_path)
        # 显示图片
        show_image(self.img_path, self.ui.imgin_win)
        # 清空输出图片
        self.ui.imgout_win.clear()
        # 刷新控件状态
        self.enable_controls()

    def detect_image(self):
        '''检测图片'''
        # 初始化调色板
        palette, pallette_dict = palette_init()
        # 转换tensor图片
        img = Image.open(self.img_path)
        # 保存原大小
        ori_w, ori_h = img.size
        img = img.resize((256, 256))
        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            img = img.to(self.device)
            outputs = self.model(img)
            out = outputs.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            out_classindex = np.unique(out)[1:]# 去除背景
            mask = Image.fromarray(out)
            mask.putpalette(palette)
            # mask.show()
            mask = mask.convert('RGB')
            # 融合图像使用的cv格式mask
            mask_cv = np.array(mask)[..., ::-1]
            # 将mask图像还原到原图大小
            mask_cv = cv2.resize(mask_cv, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)

            # 在顶部加入标签信息
            for j in out_classindex:
                label = self.classes[j - 1]
                color_rgb = pallette_dict[str(j)]  # RGB 颜色值为红色
                # 将 RGB 颜色值转换为 BGR 颜色值
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                cv2.putText(mask_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)
                self.append_text('缺陷类型：' + label)

            # 保存图片的路径
            img_out_path = os.path.join(self.output_folder, os.path.basename(self.img_path))
            print("保存位置: ", img_out_path)
            self.append_text("保存位置: "+img_out_path)

            # 输出图片的方式
            if self.method == "fusion":
                # 图像融合
                ori_img = cv2.imread(self.img_path)
                img_show = cv2.addWeighted(ori_img, 0.6, mask_cv, 0.4, 0)
            elif self.method == "mask":
                img_show = mask_cv
            # 显示图片
            self.show_imageout(img_show, self.ui.imgout_win)
            # 保存图片
            cv2.imwrite(img_out_path, img_show)

    def show_imageout(self,img_cv2, win):
        # 将OpenCV图像转换为Qt图像
        h, w, ch = img_cv2.shape
        size = win.size()
        bytes_per_line = ch * w
        img_qt = QImage(img_cv2.data, w, h, bytes_per_line, QImage.Format_BGR888)
        scaled_pixmap = img_qt.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        # 显示Qt图像
        pixmap = QPixmap.fromImage(scaled_pixmap)
        win.setPixmap(pixmap)


# 调试板初始化
def palette_init():
    palette_path = r'./utils/palette.json'
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    return pallette,pallette_dict
if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    # 展示窗口
    w.ui.show()

    app.exec()
