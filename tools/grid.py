import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def extract_grid_features(binary_image, n=5):
    # 获取图像尺寸
 
    h, w = binary_image.shape
    grid_h, grid_w = h // n, w // n
    #print(grid_h,grid_w)
    # 计算每个网格中的白色像素比例
    feature_vector = []
    for i in range(n):
        for j in range(n):
            grid = binary_image[i * grid_h: (i + 1) * grid_h, j * grid_w: (j + 1) * grid_w]
            white_pixel_ratio = np.sum(grid == 255) / (grid_h * grid_w)
            feature_vector.append(white_pixel_ratio)
    
    return np.array(feature_vector)

def gird_cosine_similarity(img1,img2,n=5):
    features1 = extract_grid_features(img1, n)
    #print(features1)
    
    features2 = extract_grid_features(img2, n)
    #print(features2)
    # 计算余弦相似度
    similarity = cosine_similarity([features1], [features2])[0][0]
    return similarity

