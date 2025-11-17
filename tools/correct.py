import cv2
import numpy as np

from PIL import Image, ImageEnhance

def correct(img,save_is=False):
    """
    绿色的没有连接起来的格子
    """
    # 读取图片并灰度化
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # 二值化处理
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    

    if save_is:
        cv2.imwrite("/home/yly/workspace/text_score/testimg/gray.png", gray)
        cv2.imwrite("/home/yly/workspace/text_score/testimg/binary.png", binary)

    # 找外轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 角点排序函数
    def sort_corners(corners):
        sorted_corners = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        sorted_corners[0] = corners[np.argmin(s)]       # 左上
        sorted_corners[2] = corners[np.argmax(s)]       # 右下
        sorted_corners[1] = corners[np.argmin(diff)]    # 右上
        sorted_corners[3] = corners[np.argmax(diff)]    # 左下

        return sorted_corners
    # 初始化最左上、右上、左下和右下的角点
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    # 遍历每个轮廓并找到四个角点
    for contour in contours:
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # 获取矩形的四个角点
        box = np.int32(box)  # 将角点转换为整数

        # 排序角点，确保是左上、右上、右下、左下的顺序
        sorted_corners = sort_corners(box)

        # 更新最左上、右上、左下和右下的角点
        if top_left is None or  sorted_corners[0][1] < top_left[1]:
            top_left = sorted_corners[0]
        
        if top_right is None or  sorted_corners[1][1] < top_right[1]:
            top_right = sorted_corners[1]
        
        if bottom_left is None or sorted_corners[3][1] > bottom_left[1]:
            bottom_left = sorted_corners[3]
        
        if bottom_right is None or  sorted_corners[2][1] > bottom_right[1]:
            bottom_right = sorted_corners[2]
        #print(top_left, top_right, bottom_right, bottom_left)
        
    sorted_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    # 计算原图角点之间的宽度和高度
    width_top = np.linalg.norm(sorted_corners[1] - sorted_corners[0])
    width_bottom = np.linalg.norm(sorted_corners[2] - sorted_corners[3])
    width = max(width_top, width_bottom)

    height_left = np.linalg.norm(sorted_corners[3] - sorted_corners[0])
    height_right = np.linalg.norm(sorted_corners[2] - sorted_corners[1])
    height = max(height_left, height_right)

    # 创建目标角点（保持长宽比）
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(sorted_corners, dst_corners)

    # 应用透视变换
    corrected_img = cv2.warpPerspective(img, M, (int(width), int(height)))
    return corrected_img

# 保存矫正后的图片
def correct_mi(img,save_is=False):
    """
    米字格
    """
   
    
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 增强对比度
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(2)  # 对比度倍数可调

    # 增强亮度
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(1.1)

    # 转为OpenCV格式（BGR）
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    

   
    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #_, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    
    if save_is:
        cv2.imwrite("/home/yly/workspace/text_score/testimg/gray.png", gray)
        cv2.imwrite("/home/yly/workspace/text_score/testimg/binary.png", binary)

    # 找外轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 角点排序函数
    def sort_corners(corners):
        sorted_corners = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        sorted_corners[0] = corners[np.argmin(s)]       # 左上
        sorted_corners[2] = corners[np.argmax(s)]       # 右下
        sorted_corners[1] = corners[np.argmin(diff)]    # 右上
        sorted_corners[3] = corners[np.argmax(diff)]    # 左下

        return sorted_corners
 
    contour = max(contours, key=cv2.contourArea)

    # 拟合多边形得到角点
    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)
    print(corners)
    if len(corners) == 4:
        corners = corners.reshape(4, 2)
    else:
        raise ValueError("未能准确找到四个角点！")

    sorted_corners = sort_corners(corners)
    # 计算原图角点之间的宽度和高度
    width_top = np.linalg.norm(sorted_corners[1] - sorted_corners[0])
    width_bottom = np.linalg.norm(sorted_corners[2] - sorted_corners[3])
    width = max(width_top, width_bottom)

    height_left = np.linalg.norm(sorted_corners[3] - sorted_corners[0])
    height_right = np.linalg.norm(sorted_corners[2] - sorted_corners[1])
    height = max(height_left, height_right)

    # 创建目标角点（保持长宽比）
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(sorted_corners, dst_corners)

    # 应用透视变换
    corrected_img = cv2.warpPerspective(img, M, (int(width), int(height)))
    return corrected_img

if __name__=='__main__':
    image_path="/data/yly/text_data/multi/test2.jpg"
    img = cv2.imread(image_path)
    corrected_img=correct_mi(img,True)
    cv2.imwrite("/home/yly/workspace/text_score/testimg/corrected_img2.png", corrected_img)
