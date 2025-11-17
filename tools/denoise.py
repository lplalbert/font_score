import cv2
import numpy as np
from PIL import Image, ImageEnhance

def enhance(image,num1=1.5,num2=2):
    image_rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 计算图像的平均亮度（灰度图）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_image)
    if avg_brightness < 140:  # 图像较暗
        num1=1.5
        num2=1.8
    elif avg_brightness <170:  # 图像较亮
        num1=1.5
        num2=1.5
    elif avg_brightness <190:
        num1=1.0
        num2=0.9
    else:
        num1=0.8
        num2=0.7
# 增强对比度
    enhancer_contrast = ImageEnhance.Contrast(image_rgb)
    image_rgb = enhancer_contrast.enhance(num1)  # 对比度倍数可调

    # 增强亮度
    enhancer_brightness = ImageEnhance.Brightness(image_rgb)
    image_rgb = enhancer_brightness.enhance(num2)


    # 转为OpenCV格式（BGR）
    image_cv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    


    return image_cv

def anti_alias(binary_img):#抗锯齿处理
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(binary_img, kernel, iterations=1)
    blurred = cv2.GaussianBlur(dilated, (3, 3), 0)
    _, anti_aliased = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    return anti_aliased

def denoise_select_green(img):
    """
    绿色田字格背景去除（专用纸张）
    变成二值化图像
    """
    # resize_to=(128,128)
    # img = cv2.resize(img, resize_to)
    enhance_img=enhance(img)
    #cv2.imwrite("/home/yly/workspace/text_score/testimg/enhance_img.png", enhance_img)
    gray = cv2.cvtColor(enhance_img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("/home/yly/workspace/text_score/testimg/gray_img.png", gray)
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    #cv2.imwrite("/home/yly/workspace/text_score/testimg/binary_img.png", binary)
    
    
    # 1. 使用中值滤波去除小噪点
    binary = cv2.medianBlur(binary, 3)  # 可能需要根据情况调整滤波器大小
   


    binary = anti_alias(binary)
    return binary
def remove_red_grid(image):
    """
    去除红色（包括深红、暗红、灰红）米字格线条 + 小像素残余
    """
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 扩展红色范围以适配更暗的红色
    lower_red1 = np.array([0, 10, 20])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 10, 20])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 膨胀操作放大红色区域
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    # 替换为白色
    image[red_mask > 0] = [255, 255, 255]

    # 中值滤波去除孤立红点
    image = cv2.medianBlur(image, 3)
    #cv2.imwrite("debug_red_mask.png", red_mask)
    return image


def denoise_select_red(img):
    """
    去除红色米字格并二值化图像
    """
    resize_to=(128,128)
    img = cv2.resize(img, resize_to)
    enhanced_img = enhance(img)
    no_red_img = remove_red_grid(enhanced_img)
    gray = cv2.cvtColor(no_red_img, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)


    binary = anti_alias(binary)
    return binary

def denoise_select_greenmi(img):
    """
    去除绿色米字格并二值化图像
    """
    pass

def denoise_select_black(img):
    """
    去除黑色格并二值化图像
    """
    pass

if __name__=='__main__':
    image_path="/home/yly/workspace/text_score/image/test2_c.jpg"
    #corrected_img=correct(image_path,True)
    img=cv2.imread(image_path)
    #result=denoise_select1(img)
    result=enhance(img)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/home/yly/workspace/text_score/testimg/result.png", result)
    cv2.imwrite("/home/yly/workspace/text_score/testimg/gray.png", gray)
    