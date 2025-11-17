import cv2
import numpy as np
import os
from tools.correct import *
import math

import cv2
import numpy as np




# 图像预处理：二值化、反转、去噪、去背景
def preprocess_image(image_path, invert=True,correct_use=True):
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if correct_use:
        # 先读取图像，然后传递给correct函数
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        corrected_img=correct(img)
        # 从原路径中提取文件名
        base_name = os.path.basename(image_path)  # 获取文件名部分（带扩展名）
        name, ext = os.path.splitext(base_name)   # 分离文件名和扩展名
        cv2.imwrite(f"temp_images/{name}_corrected{ext}", corrected_img)
        #cv2.imwrite("temp_images/corrected_img.png",corrected_img)
        image_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
        mask = image_rgb[:, :, 0] > 50  # 0代表红色通道
        # 将符合条件的区域替换为白色
        image_rgb[mask] = [255, 255, 255]
        # 将RGB格式转换回BGR格式
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"temp_images/{name}_bgr.png",image_bgr)
        corrected_img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        _, binary_img = cv2.threshold(corrected_img_gray, 100, 255, cv2.THRESH_BINARY)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary_img = cv2.resize(binary_img, (256,256))

    # 转换为灰度图像
   
   
    # cv2.imwrite("binary1.png",binary_img)
    #print(binary_img)
    if invert:
        binary_img = cv2.bitwise_not(binary_img)
        # print(binary_img)
    
    #binary_img = cv2.medianBlur(binary_img, 5)
    cv2.imwrite(f"temp_images/{name}_binary.png",binary_img)
   
    
    
    # 使用中值滤波去除噪点
    # kernel = np.ones((3,3), np.uint8)
    # kernel1 = np.ones((5,5), np.uint8)
    # cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_CLOSE, kernel1)
    # cv2.imwrite("cleaned_img.png", cleaned_img)
    denoised_img = cv2.medianBlur(binary_img, 3)  # 使用5x5的滤波器
    #denoised_img = cv2.medianBlur(denoised_img, 5)
    cv2.imwrite(f"temp_images/{name}_denoised_image.png", denoised_img)
    return denoised_img


# 图像缩放：保持长宽比
def resize_image(image, target_size=512):
    h=image.shape[0]
    w=image.shape[1]
    max_side = max(w, h)
    square_img = cv2.resize(image, (max_side, max_side))
    resized_img = cv2.resize(square_img, (target_size, target_size))
    return resized_img
# def resize_wh(image, scale_factor=3):
#     """
#     等比例缩放图像，扩大3倍。
    
#     :param image: 输入图像 (numpy 数组)
#     :param scale_factor: 缩放比例（默认为3）
#     :return: 缩放后的图像
#     """
#     h, w = image.shape[:2]  # 获取原始尺寸
#     new_w = int(w * scale_factor)
#     new_h = int(h * scale_factor)

#     # 等比例缩放
#     resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     return resized_img





def crop_and_resize(image, bbox, target_size=(128,128)):
    """ 从图像中裁剪 bbox 指定的部分(字的外边距），并缩放到 target_size
    返回缩放后图，面积，长宽比
    """
    x_min, y_min, x_max, y_max = bbox
    
    # 1. 裁剪区域
    cropped_region = image[y_min:y_max, x_min:x_max]
    
    # 2. 计算面积
    width, height = x_max - x_min, y_max - y_min
    area = width * height
    wh_scale=width/height

    # 3. 缩放到目标尺寸
    resized_region = cv2.resize(cropped_region, target_size, interpolation=cv2.INTER_AREA)
    #cv2.imwrite("crop.png",resized_region)

    return  resized_region, area, wh_scale


def crop(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    
    # 1. 裁剪区域
    cropped_region = image[y_min:y_max, x_min:x_max]
    
    # 2. 计算面积
    width, height = x_max - x_min, y_max - y_min
    area = width * height

    return  cropped_region, area



# 字符图像平移：计算图像偏移
def shift_image(image, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted_image

def find_max_text_bbox(image):
    """
    输入二值图像，得到最小外边距和凸包
    """
    # 确保输入为二值图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  # 形态学优化（保持结构完整性）
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    # processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 寻找轮廓（假设文字为白色255）
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    
    if not contours:
        return None  # 无文字区域
    # 遍历每个轮廓，计算其最小外接矩形
   # 初始化最外围矩形的坐标
    x_min, y_min, w_max, h_max = float('inf'), float('inf'), 0, 0

    # 遍历每个轮廓，计算最小外接矩形
    for contour in contours:
        # 获取每个轮廓的最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 更新最外围矩形的坐标
        x_min = min(x_min, x)  # 左边界
        y_min = min(y_min, y)  # 上边界
        w_max = max(w_max, x + w)  # 右边界
        h_max = max(h_max, y + h)  # 下边界

    # 找到最大面积的轮廓
    #print(len(contours))
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算外接矩形
    #x, y, w, h = cv2.boundingRect(max_contour)
    all_contours = np.concatenate(contours)
    hull = cv2.convexHull(all_contours)  # 假设只有一个轮廓
    return (x_min, y_min, w_max, h_max),hull  # 返回左上和右下坐标

def normalized_cos(x):

    return (x+1)/2

def is_chinese(char):
    # 检查是否为汉字的Unicode块
    return any([
        '\u4e00' <= char <= '\u9fff',     # 常用汉字
        '\u3400' <= char <= '\u4dbf',     # 扩展A
        '\u20000' <= char <= '\u2a6df',   # 扩展B
        '\u2a700' <= char <= '\u2b73f',   # 扩展C
        '\u2b740' <= char <= '\u2b81f',   # 扩展D
        '\u2b820' <= char <= '\u2ceaf',   # 扩展E
        '\u2ceb0' <= char <= '\u2ebef',   # 扩展F
        '\u30000' <= char <= '\u3134f'    # 扩展G
    ])



if __name__=="__main__":
    img = cv2.imread('/home/yly/workspace/text_score/testimg/cat.png', cv2.IMREAD_GRAYSCALE)  # 灰度读入
    bbox,hull=find_max_text_bbox(img)
    print(bbox)