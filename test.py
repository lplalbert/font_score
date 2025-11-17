from tools import *


if __name__=="__main__":

    source_image_path = r'/home/yly/workspace/text_score/data/di/demo1.png'  # 替换为实际路径
    target_image_path = r'/home/yly/workspace/text_score/data/di/demo2.png'  # 替换为实际路径
    src_img = preprocess_image(source_image_path)
    target_img = preprocess_image(target_image_path)
    src_char_img_resized = resize_image(src_img, 128)
    target_char_img_resized = resize_image(target_img , 128)

    # 计算平移量（通过计算重心的偏移来确定）
    cX_src, cY_src = calculate_center_of_gravity(src_char_img_resized)
    cX_target, cY_target = calculate_center_of_gravity(target_char_img_resized)
    dx = cX_target - cX_src
    dy = cY_target - cY_src

    bbox_s,hull_s = find_max_text_bbox(src_char_img_resized)
    bbox_t,hull_t = find_max_text_bbox(target_char_img_resized)

    #裁剪出字符区域，以及字符面积
    cropped_s, area_s = crop_and_resize(src_char_img_resized, bbox_s)
    cropped_t, area_t = crop_and_resize(target_char_img_resized, bbox_t)
    area_dif=np.abs(area_t-area_s)
    #细化
    thin_image_s = cv2.ximgproc.thinning(cropped_s) 
    thin_image_t = cv2.ximgproc.thinning(cropped_t) 

    hu_score=calculate_weighted_sum(cropped_s, cropped_t)#笔画相似度
    similar_layout_score = calculate_similar_layout(thin_image_s,thin_image_t) #布局
    vgg_score=features_similarity(thin_image_s,thin_image_t) #vgg特征

    

    #计算相关系数
    match_rate = cv2.matchTemplate(cropped_s, cropped_t, cv2.TM_CCOEFF_NORMED)

    # #计算最大覆盖率
    # coverage_rate=max_coverage_rate(cropped_s, cropped_t)

    hull_handwritten_norm=normalize_convex_hull(hull_s, target_size=(50, 50))
    hull_template_norm=normalize_convex_hull(hull_t, target_size=(50, 50))

    # 计算相似性指标
    similarity = convex_hull_similarity(hull_template_norm, hull_handwritten_norm)
    distance = hausdorff_distance(hull_template_norm, hull_handwritten_norm)
    area_similarity, perimeter_similarity, aspect_ratio_similarity = area(hull_template_norm, hull_handwritten_norm)

    #print(dx,dy,area_dif,match_rate,similarity,distance,area_similarity,perimeter_similarity,aspect_ratio_similarity)
    # 评分
    # 计算偏移量的评分
    offset_score = max(0, 100 - (np.sqrt(dx**2 + dy**2) * 1))
    # 计算面积差异的评分
    area_diff_score = max(0, 100 - area_dif * 0.01)
    # 计算相似度的评分
    coverage_score = match_rate[0][0]*100  # 假设覆盖率越高越好
    # 计算凸包相似度的评分
    hull_similarity_score = similarity * 100  # 相似度越高，评分越高
    # 计算 Hausdorff 距离的评分
    hausdorff_score = max(0, 100 - distance)  # 距离越小，评分越高
    # 计算形状特征相似度的评分
    shape_feature_score = (area_similarity + perimeter_similarity + aspect_ratio_similarity) / 3 * 100

   
    similar_layout_score=similar_layout_score*100
    print(vgg_score)
    vgg_score=100-vgg_score*100

    print(hull_similarity_score ,similar_layout_score,shape_feature_score,hausdorff_score)

    print(hull_similarity_score * 0.2 ,similar_layout_score*0.1,shape_feature_score*0.1,hausdorff_score * 0.1 )
    print("vgg:",vgg_score)
    print("area",area_diff_score)
    print("offset",offset_score)
    final_score = (
        hull_similarity_score * 0.2 + 
        vgg_score*0.1+
        similar_layout_score*0.1+
        hausdorff_score * 0.1 + 
        shape_feature_score*0.1+
        area_diff_score * 0.2 + 
        offset_score * 0.2     
    )

    print(f"最终评分: {final_score:.2f} / 100")
    print("详细评价：")
    shape_score=hull_similarity_score * 0.2 + similar_layout_score*0.1+shape_feature_score*0.1+hausdorff_score * 0.1 
    print(shape_score)

    if shape_score < 30:
        print( '字形极其不相似')
    elif shape_score < 40:
        print('字形很不相似') 
    elif shape_score <47:
        print('字形基本相似')
    else:
        print('字形非常相似')
    
    if vgg_score < 20:
        print( '骨架极其不相似')
    elif vgg_score < 40:
        print('骨架很不相似') 
    elif vgg_score < 60:
        print ('骨架不太相似')
    elif vgg_score < 80:
        print('骨架位置基本相似')
    else:
        print ('骨架位置非常相似')

    if area_diff_score < 20:
        print( '字体大小极其不相似')
    elif area_diff_score < 60:
        print('字体大小很不相似') 
    elif area_diff_score < 90:
        print('字体大小基本相似')
    else:
        print ('字体大小非常相似')
    
    if offset_score < 20:
        print( '位置极其不相似')
    elif offset_score < 40:
        print('位置很不相似') 
    elif offset_score < 80:
        print ('位置不太相似')
    elif offset_score < 90:
        print('位置基本相似')
    else:
        print ('位置非常相似')
   
