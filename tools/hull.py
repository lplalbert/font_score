import cv2
from shapely.geometry import Polygon
import numpy as np

# 计算凸包
def convex_hull(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])  # 假设只有一个轮廓
    

    return hull
def convex_hull_similarity(hull1, hull2):
    """ 计算两个凸包的 Jaccard 相似度 """
    poly1 = Polygon(hull1[:, 0, :])  # 转换为 Shapely 多边形
    poly2 = Polygon(hull2[:, 0, :])
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    jaccard_similarity = intersection_area / union_area
    return jaccard_similarity

def hausdorff_distance(hull1, hull2):
    """ 计算 Hausdorff 距离 """
    return cv2.createHausdorffDistanceExtractor().computeDistance(hull1, hull2)

def convex_hull_features(hull):
    """ 计算凸包的形状特征 """
    area = cv2.contourArea(hull)  # 计算面积
    perimeter = cv2.arcLength(hull, True)  # 计算周长
    x, y, w, h = cv2.boundingRect(hull)  # 获取外接矩形
    aspect_ratio = w / h  # 计算长宽比
    return area, perimeter, aspect_ratio
def area(hull1,hull2):
    # 计算手写字和模板字的特征
    area1, perimeter1, aspect_ratio1 = convex_hull_features(hull1)
    area2, perimeter2, aspect_ratio2 = convex_hull_features(hull2)

    # 计算相似度 面积 周长 长宽比
    area_similarity = min(area1, area2) / max(area1, area2)
    perimeter_similarity = min(perimeter1, perimeter2) / max(perimeter1, perimeter2)
    aspect_ratio_similarity = min(aspect_ratio1, aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
    return area_similarity,perimeter_similarity,aspect_ratio_similarity

def normalize_convex_hull(hull, target_size=(100, 100)):
    """ 归一化凸包到相同尺寸（保持比例） """
    x, y, w, h = cv2.boundingRect(hull)  # 获取外接矩形
    scale = min(target_size[0] / w, target_size[1] / h)  # 计算缩放比例（保持比例）
    
    # 计算偏移量（让凸包居中）
    offset_x = (target_size[0] - w * scale) / 2
    offset_y = (target_size[1] - h * scale) / 2

    # 缩放并居中凸包
    normalized_hull = np.array(
        [[(int((pt[0][0] - x) * scale + offset_x), int((pt[0][1] - y) * scale + offset_y))] for pt in hull],
        dtype=np.int32
    )

    return normalized_hull
