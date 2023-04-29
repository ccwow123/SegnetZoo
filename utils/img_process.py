import cv2
import numpy as np

''' 
原理：
cv找到出电容的轮廓，然后统计出其中的总像素cap_area
然后根据针孔的坐标找到针孔的轮廓，然后做同样的处理，统计出总像素target_area 
自动电容端面的标准面积reference_area
就可以获得针孔面积true_area=reference_area*(target_area /cap_area)



操作：
1.准备好缺陷电容图片
2.准备好缺陷的mask图片（labelme标注好后，实验voc的数据集转换脚本可生成在data_dataset_voc\SegmentationClass内）
3.改图片路径
'''


# 获取轮廓
def get_contour(img_in, threshold1=0, threshold2=25, bin_parse=74,dilate_iterations=16, erode_iterations=0):
    '''

    :param img_in: 输入图像
    :param threshold1: canny第一个阈值
    :param threshold2: canny第二个阈值
    :param bin_parse: 二值化阈值
    :param dilate_iterations:腐蚀重复次数
    :param erode_iterations:膨胀重复次数
    :return:
    '''
    imgGray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    ret, imgBin = cv2.threshold(imgGray, bin_parse, 255, cv2.THRESH_BINARY) # 二值化
    imgThreshold = cv2.Canny(imgBin, threshold1, threshold2)  # 边缘检测器
    kernel = np.ones((5, 5))  # 图像处理的卷积核
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=dilate_iterations)  # 图像处理：膨胀
    imgThreshold = cv2.erode(imgDial, kernel, iterations=erode_iterations)  # 图像处理：腐蚀  膨胀腐蚀可以帮助我们消除缝隙和杂物
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

    return contours


def get_target_true_area(path, path2, reference_width=1.3225, reference_height=1.3225):
    '''

    :param path: 原图的路径
    :param path2: mask的路径
    :param reference_width: 参考物的宽
    :param reference_height: 参考物的长
    :return: 目标真实面积
    '''
    # 设定参照物尺寸 单位：mm
    # reference_width = 1.3225
    # reference_height = 1.3225
    reference_area = reference_width * reference_height  # 前提参照物是矩形
    # 读取图片和针孔mask （0.5缩放）
    img_in = path
    # size_x, size_y = img.shape[0:2]
    # img_in = cv2.resize(img, (int(size_y / 2), int(size_x / 2)))
    img2_in = path2
    # size_x, size_y = img2.shape[0:2]
    # img2_in = cv2.resize(img2, (int(size_y / 2), int(size_x / 2)))
    # 找端面轮廓
    cap_contours = get_contour(img_in, 0, 180)
    # img_out = cv2.drawContours(img_in, cap_contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
    # 获得参照物的像素
    cap_area = cv2.contourArea(cap_contours[0])
    # 获得参考比例
    reference_rate = reference_area / cap_area
    # reference_rate = cap_area / reference_area  # 或者分子分母反过来
    # 获得缺陷目标的总像素
    target_contours = get_contour(img2_in, 0, 15)
    # target_contours=path2
    box_list = []  # 外接矩形坐标list
    true_area_list = []  # 各个部分面积list
    for i, c in enumerate(target_contours):
        # 计算各个部分面积
        target_area = cv2.contourArea(target_contours[i])
        # true_area = target_area / reference_rate
        true_area = reference_rate * target_area
        if true_area > 0.0001:
            true_area_list.append('%.5f' % true_area)
            x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            box_list.append([x, y])
        print(f'{i + 1}号缺陷面积为：%.5f mm2' % true_area)
    # target_contours = get_contour(img2_in, 0, 15)
    # target_area = cv2.contourArea(target_contours[0])
    # # 计算出缺陷的真实面积
    # true_area = target_area / reference_rate
    # print('缺陷面积为：%.5f mm2' % true_area)
    return true_area_list, box_list, target_contours


def get_target_true_area2(path, path2, reference_width=1.2, reference_height=2.0):
    '''
    :param path: 原图的路径
    :param path2: mask的路径
    :param reference_width: 参考物的宽
    :param reference_height: 参考物的长
    :return: 目标真实面积
    '''
    # 设定参照物尺寸 单位：mm
    reference_area = reference_width * reference_height  # 前提参照物是矩形
    # 读取图片和针孔mask
    img_in = path
    img2_in = path2
    # 找端面轮廓
    cap_contours = get_contour(img_in,20, 75)  # 最好先找到端面的轮廓的阈值
    # 获得参照物的像素
    cap_area=0
    for _ ,sub_contours in enumerate(cap_contours):
        temp = cv2.contourArea(sub_contours)
        if temp>cap_area:
            cap_area=temp
    # cap_area = cv2.contourArea(cap_contours[0])
    # 获得参考比例
    reference_rate = reference_area / cap_area
    print('参考比值：%r' % reference_rate)
    # 获得缺陷目标的总像素
    target_contours, _ = cv2.findContours(img2_in, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图中将预测的地方用红色边框框出来
    result_img = cv2.drawContours(img_in, target_contours, -1, (0, 0, 255), 3)

    # 面积信息处理
    box_list = []  # 外接矩形坐标list
    true_area_list = []  # 各个部分面积list
    area_scale_list = []  # 各个部分面积占比list
    i2 = 0
    for i, c in enumerate(target_contours):
        # 计算各个部分面积
        target_area = cv2.contourArea(c)
        true_area = reference_rate * target_area
        if true_area > 0.001:
            i2 += 1
            true_area_list.append('%.3f' % true_area)
            # 计算面积百分比
            scale = true_area / reference_area * 100
            area_scale_list.append(round(scale, 2))
            # 获得面积坐标
            x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            box_list.append([x, y])
            # 显示函数
            print(f'{i2}号缺陷面积为：%.3f mm2，占比%.2f%%' % (true_area, scale))

    return true_area_list, area_scale_list, box_list, result_img


# 找到预测图中白色轮廓信息并进行处理
def contours_process(origin_img, pred_img,label):
    # 重构
    # 获取缺陷目标面积和位置
    target_area, area_scale_list, box_list, result_img = get_target_true_area2(origin_img, pred_img)
    # 显示面积文字
    for i2 in range(len(target_area)):
        # area_show=target_area[i2] + 'mm2'#面积文字
        scale_show = str(area_scale_list[i2]) + '%'  # 面积占比文字
        info = label+scale_show
        position = (box_list[i2])

        # 不想显示面积占比的话，把下面这行注释掉
        #                           图片     添加的文字   位置      字体                字体大小 字体颜色 字体粗细
        result_img = cv2.putText(result_img, info, position, cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 0, 255), 2)#面积文字

    return result_img

# 找到预测图中白色轮廓信息并进行处理
def contours_process_nolabel(origin_img, pred_img):
    # 重构
    # 获取缺陷目标面积和位置
    target_area, area_scale_list, box_list, result_img = get_target_true_area2(origin_img, pred_img)
    # 显示面积文字
    for i2 in range(len(target_area)):
        # area_show=target_area[i2] + 'mm2'#面积文字
        scale_show = str(area_scale_list[i2]) + '%'  # 面积占比文字
        info = scale_show
        position = (box_list[i2])

        # 不想显示面积占比的话，把下面这行注释掉
        #                           图片     添加的文字   位置      字体                字体大小 字体颜色 字体粗细
        result_img = cv2.putText(result_img, info, position, cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 0, 255), 2)#面积文字

    return result_img
if __name__ == '__main__':
    path = r'D:\Files\_datasets\voc_self\data_dataset_voc\JPEGImages\E_pinhole_17.jpg'  # 电容图片
    path2 = r'D:\Files\_datasets\voc_self\data_dataset_voc\SegmentationClass\E_pinhole_17.png'  # 缺陷mask图片   注意路径不能有中文
    # get_target_true_area(path,path2)
    # path2=get_contour(path2,0, 25)
    # cnts = cv2.findContours(path2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # ==获取缺陷目标面积
    target_area, box_list, cnts = get_target_true_area(path, path2)
    # print('缺陷面积为：%.5f mm2' % target_area)

    img_in = cv2.imread(path)
    img2_in = cv2.imread(path2)
    # 在原图中将预测的地方用红色边框框出来
    result_img = cv2.drawContours(img_in, cnts, -1, (255, 0, 0), 1)
    # ==显示面积文字
    for i2 in range(len(target_area)):
        result_img = cv2.putText(result_img, target_area[i2] + 'mm2', (box_list[i2]), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                 (255, 0, 0), 2)
    # 保存图片
    save_path = r'C:\Users\18493\Desktop\tt2\11'
    print(save_path + '/' + "2.jpg")
    cv2.imwrite(save_path + '/' + "2.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
