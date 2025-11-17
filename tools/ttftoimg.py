from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import sys
sys.path.append('/home/ylyu/workspace/text_score')
from tools.util import crop_and_resize, find_max_text_bbox

def render_text_to_cv(ttf_path, text, font_size=100, img_size=(256,256)):
    """
    渲染文字到 OpenCV 格式的图片 (numpy数组)
    :param ttf_path: 字体路径
    :param text: 要渲染的文字
    :param font_size: 字体大小
    :param img_size: 整体图片大小 (宽, 高)，可选
    :return: OpenCV风格的numpy数组 (gray)
    """
    # 加载字体
    font = ImageFont.truetype(ttf_path, font_size)

    # 创建透明背景的Pillow图像
    if img_size is None:
        img_size = (font_size * 2, font_size * 2)
    pil_img = Image.new('RGBA', img_size, (0, 0, 0, 255))

    # 画字
    draw = ImageDraw.Draw(pil_img)
    
    # bbox = draw.textbbox((0, 0), text, font=font)
    bbox = font.getbbox(text)
    #print(bbox)
    #print(bbox)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    #text_width, text_height = font.getsize(text)
    x = (img_size[0] - text_width) // 2
    y = max((img_size[1] - text_height) // 2-bbox[1],0)
    #print("x,y:",text_width,text_height,x,y)
    draw.text((x,y), text, font=font, fill=(255, 255, 255, 255))

    # Pillow (RGBA) -> numpy array
    cv_img = np.array(pil_img)

    # 把RGBA转成OpenCV常用的BGR
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    return cv_img

# ==== 示例使用 ====
if __name__=='__main__':
    ttf_path = '/mnt/ylyu/text_data/ttf_tar/华栋正楷第三版 Regular.ttf'
    text = '一'
    font_size = 120
    img_cv = render_text_to_cv(ttf_path, text, font_size)
    bbox_t,hull_t = find_max_text_bbox(img_cv)
    cropped_t, area_t,wh_scalet = crop_and_resize(img_cv, bbox_t)
    print(area_t)
    print(wh_scalet)

    # 也可以直接用cv2.imwrite保存
    cv2.imwrite('/home/ylyu/workspace/text_score/testimg/ni.png', img_cv)
    cv2.imwrite('/home/ylyu/workspace/text_score/testimg/cropped_t.png', cropped_t)
