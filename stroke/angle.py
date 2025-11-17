import numpy as np
import cv2
def analyze_stroke_tilt(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    
    if not contours:
        print("contours none")
        return None  # 无文字区域
    
    # 找到最大面积的轮廓
    #print(len(contours))
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    (x,y), (w,h), angle = rect  
    actual_angle = angle + 90 if w < h else angle
    return actual_angle

def angle_dif(img1,img2):
    angle1=analyze_stroke_tilt(img1)
    angle2=analyze_stroke_tilt(img2)
    #print(angle1,angle2)
    angle_d=np.abs(angle1-angle2)/90
    
    return angle_d
