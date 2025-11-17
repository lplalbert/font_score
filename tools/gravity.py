import cv2
import math

# 计算重心
def calculate_center_of_gravity(image):
    moments = cv2.moments(image)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        cX, cY = 0, 0
    return cX,cY
# 计算单张图像的重心偏离中心的得分
def gravity_offset_single(image):
    """
    单张图计算重心
    """
    h, w = image.shape[:2]
    center_x = w // 2
    center_y = h // 2

    cX, cY = calculate_center_of_gravity(image)
    #print(cX,cY,center_x,center_y)

    dx = center_x - cX
    dy = center_y - cY

    deviation = math.sqrt(dx ** 2 + dy ** 2)
    max_deviation = math.sqrt(center_x ** 2 + center_y ** 2)  # 最大可能偏移量

    offset_score = deviation / max_deviation * 100
    offset_score = max(0, 100 - offset_score)

    return offset_score
def get_min_bounding_box_single(bboxs,image):
    """
    单张图计算外边距中心
    """
    h, w = image.shape[:2]
    center_x = w // 2
    center_y = h // 2

    left1, bottom1,right1,top1 = bboxs
    
    # 计算中心点
    center1_x = (left1 + right1) / 2
    center1_y = (top1 + bottom1) / 2
    #print(center1_x,center1_y,center_x,center_y)
   
    # 计算中心点偏差（欧几里得距离）
    deviation = math.sqrt((center_x - center1_x) ** 2 + (center_y - center1_y) ** 2)
    max_deviation = math.sqrt(center_x ** 2 + center_y ** 2)  # 最大可能偏移量

    offset_score = deviation / max_deviation * 100
    offset_score = max(0, 100 - offset_score)

    return offset_score


#需要src和target
def gravity_offset(src_char_img_resized,target_char_img_resized):
    # 计算平移量（通过计算重心的偏移来确定）
    cX_src, cY_src = calculate_center_of_gravity(src_char_img_resized)
    cX_target, cY_target = calculate_center_of_gravity(target_char_img_resized)
    dx = cX_target - cX_src
    dy = cY_target - cY_src
    #offset_score = max(0, 100 - (np.sqrt(dx**2 + dy**2) * 1))
    deviation = math.sqrt(dx ** 2 + dy ** 2)
    offset_score = deviation/math.sqrt(96 ** 2 + 96 ** 2)*100
    offset_score=max(0,100-offset_score)
    return offset_score

def get_min_bounding_box(bboxs,bboxt):

    left1, bottom1,right1,top1 = bboxs
    left2, bottom2,right2,top2 = bboxt
    # 计算中心点
    center1_x = (left1 + right1) / 2
    center1_y = (top1 + bottom1) / 2
    center2_x = (left2 + right2) / 2
    center2_y = (top2 + bottom2) / 2

    # 计算中心点偏差（欧几里得距离）
    deviation = math.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
    score = deviation/math.sqrt(96 ** 2 + 96 ** 2)*100
    score=max(0,100-score)
    return score

if __name__=="__main__":
    img = cv2.imread('/home/yly/workspace/text_score/testimg/cat.png', cv2.IMREAD_GRAYSCALE)  # 灰度读入
    score = gravity_offset_single(img)
    print("重心对齐得分：", score)
    score=get_min_bounding_box_single((69, 107, 188, 227),img)
    print("重心对齐得分：", score)
    
