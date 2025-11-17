import cv2
import numpy as np

#计算布局相似度，直接对原图进行Contour_extraction函数进行轮廓提取，在使用calculate_similar_layout函数进行计算



def calculate_similar_layout(img1_binary, img2_binary):

    # 计算图像中的轮廓
    contours1, _ = cv2.findContours(img1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个轮廓的布局特征
    layout_features1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x / img1_binary.shape[1]  # 归一化到[0, 1]
        right_distance = (img1_binary.shape[1] - (x + w)) / img1_binary.shape[1]
        top_distance = y / img1_binary.shape[0]
        bottom_distance = (img1_binary.shape[0] - (y + h)) / img1_binary.shape[0]
        layout_features1.append([left_distance, right_distance, top_distance, bottom_distance, w / img1_binary.shape[1],
                                 h / img1_binary.shape[0]])  # 添加字宽和字高

    layout_features2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x / img2_binary.shape[1]
        right_distance = (img2_binary.shape[1] - (x + w)) / img2_binary.shape[1]
        top_distance = y / img2_binary.shape[0]
        bottom_distance = (img2_binary.shape[0] - (y + h)) / img2_binary.shape[0]
        layout_features2.append(
            [left_distance, right_distance, top_distance, bottom_distance, w /img2_binary.shape[1], h / img2_binary.shape[0]])

        # 计算布局相似度
    layout_similarities = []
    for feature1 in layout_features1:
        max_similarity = 0
        for feature2 in layout_features2:
            distance = np.sqrt(np.sum(np.square(np.array(feature1[:-2]) - np.array(feature2[:-2]))))  # 忽略字宽和字高
            similarity = 1 / (1 + distance)  # 简化的相似度度量
            if similarity > max_similarity:
                max_similarity = similarity
        layout_similarities.append(max_similarity)

        # 计算平均相似度
    return np.mean(layout_similarities)
   
#
# # 图片路径
# img_path1 = r'E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png'
# img_path2 = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\3.png'
#
# # 调用函数计算布局相似度
# similar_layout_score = calculate_similar_layout(img_path1, img_path2)
# print("布局相似度得分为：", similar_layout_score)