import config as c
import cv2
import numpy as np
import os
from tools.correct import *
import math
from PIL import Image, ImageEnhance
import cv2
import numpy as np

def enhance(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 增强对比度
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(2.3)  # 对比度倍数可调

    # 增强亮度
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(2)

    # 转为OpenCV格式（BGR）
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 处理白色区域：变得更白（通过HSV掩码方式）
    # hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0, 0, 180])
    # upper_white = np.array([180, 50, 255])
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # image_cv[mask_white > 0] = [255, 255, 255]

    # # 保存处理后的图像
    # output_path = 'processed_image.png'
    # cv2.imwrite(output_path, image_cv)

    # print("处理完成，保存为:", output_path)
    return image_cv
def extract_black_region(img, max_rgb=30, max_diff=2):
    """
    根据RGB范围提取黑色区域掩码
    参数：
        img: 输入BGR格式图像（OpenCV默认）
        max_rgb: RGB最大值阈值，默认100
        max_diff: 最大通道和最小通道允许差值，默认50
    返回：
        mask: 黑色区域二值掩码（255为黑色区域，0为非黑）
    """
    # 拆通道，OpenCV默认BGR格式
    b, g, r = cv2.split(img)

    # 创建掩码条件：每个通道都小于max_rgb
    #cond_dark = (b > max_rgb) & (g > max_rgb) & (r > max_rgb)
    cond_dark = (r > max_rgb)
    # 最大值和最小值差异，用来排除彩色深色
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    cond_diff = (max_channel - min_channel) > max_diff

    # 综合两个条件
    #mask = cond_dark & cond_diff
    mask = cond_dark& cond_diff
    img[mask] = [255, 255, 255]  # 将黑色区域替换为白色
    #img[~mask]=[0,0,0]  # 将非黑色区域替换为黑色

    

    return img
def denoise_select1(img):
    """
    红色田字格背景去除 
    变成二值化图像
    """
    
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = image_rgb[:, :, 0] > 50  # 0代表红色通道
    # 将符合条件的区域替换为白色
    image_rgb[mask] = [255, 255, 255]
    cv2.imwrite(f"temp_images/A_dedd.png",image_rgb)



    # 第四步（可选）：再次增强黑字对比度
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    #binary_img = cv2.bitwise_not(binary)
    cv2.imwrite(f"temp_images/A_gray.png",binary)
    denoised_img = cv2.medianBlur(binary, 3)  # 使用3x3的滤波器
    #cv2.imwrite(f"temp_images/denoised_image.png", denoised_img)
    return denoised_img
image_path = c.image_path  # 田字格图片
target_ttf = c.target_ttf  # 替换为target.png文件的路径
rows = c.rows  # 例如10行
cols = c.cols   # 例如8列
resize_to = (128,128)  # 统一resize成256*256像素


# 遍历文件夹中的所有图片（排除target.png）
score_history=[]
corrected_img=correct(image_path)
cv2.imwrite(f"temp_images/A_correct.png",corrected_img)
enhance_img=enhance(corrected_img)
cv2.imwrite(f"temp_images/A_enhance_img.png",enhance_img)
denoise_img=extract_black_region(enhance_img)


cv2.imwrite(f"temp_images/A_de.png",denoise_img)