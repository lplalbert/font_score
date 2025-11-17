from tools import *
from stroke import *
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import logging
from datetime import datetime
# from test_style import style_score  # 暂时禁用需要 GPU 的风格评分模块
# 获取当前时间戳
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 创建一个以时间戳命名的日志文件
log_filename = f"log/output_{timestamp}.log"
# 配置日志
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

def main(source_image_path,target_image_path):
    base_name = os.path.basename(source_image_path)  # 获取文件名部分（带扩展名）
    names, ext = os.path.splitext(base_name)   # 分离文件名和扩展名
    base_name = os.path.basename(target_image_path)  # 获取文件名部分（带扩展名）
    namet, ext = os.path.splitext(base_name)   # 分离文件名和扩展名

    src_img = preprocess_image(source_image_path,correct_use=True)
    target_img = preprocess_image(target_image_path,correct_use=True)
    #重心
    # gravity_similarity=centroid_cosine_similarity(src_img,target_img)
    # gravity_similarity=normalized_cos(gravity_similarity)*100
    gravity_similarity=gravity_offset(src_img,target_img)
    


    bbox_s,hull_s = find_max_text_bbox(src_img)
    bbox_t,hull_t = find_max_text_bbox(target_img)
    bounding_score=get_min_bounding_box(bbox_s,bbox_t)
    gravity_score=gravity_similarity*0.7+bounding_score*0.3
    gravity_score = min(100, gravity_score ** 2 / 90)

    # result1 = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(result1, [hull_s], -1, (0, 255, 0), 2)  # 在图像1上绘制凸包
    # cv2.imwrite("temp_images/hulls.png",result1)
    # result2 = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(result2, [hull_t], -1, (0, 255, 0), 2)  # 在图像2上绘制凸包
    # cv2.imwrite("temp_images/hullt.png",result2)

    #裁剪出字符区域，以及字符面积
    cropped_s, area_s,wh_scales = crop_and_resize(src_img, bbox_s)
    cropped_t, area_t,wh_scalet = crop_and_resize(target_img, bbox_t)
    cropped_snot=cv2.bitwise_not(cropped_s) 
    cropped_tnot=cv2.bitwise_not(cropped_t) 
    cv2.imwrite(f"temp_images/{names}_crops.png",cropped_s)
    cv2.imwrite(f"temp_images/{namet}_cropt.png",cropped_t)
    cv2.imwrite(f"temp_images/{names}_cropsnot.png",cropped_snot)
    cv2.imwrite(f"temp_images/{namet}_croptnot.png",cropped_tnot)
    #area_dif=100-(np.abs(area_t-area_s)/area_t)*100
    ratio = min(area_s / area_t, area_t / area_s)
    area_dif = 100 * ratio
    #wh_dif=100-(np.abs(wh_scalet-wh_scales)/wh_scalet)*100
    ratio = min(wh_scalet / wh_scales, wh_scales / wh_scalet)
    wh_dif = 100 * ratio
   
    shape_score=area_dif*0.7+wh_dif*0.3


    # 字体风格评分（需要 GPU）暂时禁用
    style_scores = 0


    #网格特征
    grid_cosine_raw = gird_cosine_similarity(cropped_s,cropped_t,n=16)  # 原始余弦相似度 [-1, 1]
    grid_similarity_normalized = normalized_cos(grid_cosine_raw)  # 归一化到 [0, 1]
    grid_similarity_scaled = grid_similarity_normalized * 100  # 缩放到 [0, 100]
    grid_similarity = min(100, grid_similarity_scaled ** 2 / 90)  # 平方后除以90，限制最大值为100
    
    # 调试输出（可选，用于检查为什么是100）
    # print(f"DEBUG: grid_cosine_raw={grid_cosine_raw:.4f}, normalized={grid_similarity_normalized:.4f}, scaled={grid_similarity_scaled:.4f}, final={grid_similarity:.4f}")
    #骨架相似度
    thin_image_s = cv2.ximgproc.thinning(cropped_s) 
    thin_image_t = cv2.ximgproc.thinning(cropped_t)
    thin_image_s=cv2.bitwise_not(thin_image_s)
    thin_image_t=cv2.bitwise_not(thin_image_t)
    
    cv2.imwrite(f"temp_images/{names}_thins.png",thin_image_s)
    cv2.imwrite(f"temp_images/{namet}_thint.png",thin_image_t)
    vgg_score=features_similarity(thin_image_s,thin_image_t) #vgg特征
    vgg_score=100-min(1, vgg_score ** 2 / 0.7)*100
    
    ssim_score, _ = ssim(thin_image_s, thin_image_t, full=True)
    ssim_score=min(1, ssim_score ** 2 / 0.55)*100
   
    
    # 提取非零像素点位置
    pts1 = np.column_stack(np.where(thin_image_s == 0))
    pts2 = np.column_stack(np.where(thin_image_t == 0))
    # 计算Hausdorff距离
    hausdorff_dist = max(directed_hausdorff(pts1, pts2)[0], directed_hausdorff(pts2, pts1)[0])
    normalized_dist = hausdorff_dist / np.sqrt(128**2 + 128**2)
    hausdorff_dist_score = (1 - normalized_dist) * 100
    hausdorff_dist_score = max(hausdorff_dist_score, 0) # 保证不为负数
    
    skeleton_score=vgg_score*0.3+ssim_score*0.4+hausdorff_dist_score*0.3
    #skeleton_score = min(100, skeleton_score ** 2 / 100)


    #计算凸包相似度
    hull_handwritten_norm=normalize_convex_hull(hull_s, target_size=(64,64))
    hull_template_norm=normalize_convex_hull(hull_t, target_size=(64,64))
    similarity = convex_hull_similarity(hull_template_norm, hull_handwritten_norm)
    distance = hausdorff_distance(hull_template_norm, hull_handwritten_norm)
    # 计算 Hausdorff 距离的评分
    normalized_distance = distance / np.sqrt(64**2 + 64**2)
    distance_score = (1 - normalized_distance) * 100
    distance_score = max(distance_score, 0) # 保证不为负数
    # 计算凸包相似度的评分
    hull_similarity_score = similarity * 100  # 相似度越高，评分越高
    hull_score=(hull_similarity_score*0.6+distance_score*0.4)
    # print("gravity",gravity_similarity,bounding_score)
    # print("area,wh",area_dif,wh_dif)
    # print("网格：",grid_similarity)
    # print("骨架alex：",vgg_score)
    # print("SSIM相似度：", ssim_score)
    # print("Hausdorff 距离：", hausdorff_dist_score)
    # print("凸包：",hull_similarity_score,distance_score)
    # 记录到日志
    logging.info(f"gravity {gravity_similarity} {bounding_score} {gravity_score}")
    logging.info(f"area,wh {area_dif} {wh_dif} {shape_score}")
    logging.info(f"网格： {grid_similarity}")
    logging.info(f"alex： {vgg_score}")
    logging.info(f"SSIM相似度： {ssim_score}")
    logging.info(f"Hausdorff 距离： {hausdorff_dist_score}")
    logging.info(f"骨架得分：{skeleton_score}")
    logging.info(f"凸包： {hull_similarity_score} {distance_score} {hull_score}")
    logging.info(f"风格：{style_scores}")
    
    # 在控制台输出各项评价指标
    print("\n" + "=" * 60)
    print("评价指标详情")
    print("=" * 60)
    print(f"【重心评分】")
    print(f"  重心偏移相似度: {gravity_similarity:.2f}")
    print(f"  边界框相似度: {bounding_score:.2f}")
    print(f"  重心综合得分: {gravity_score:.2f}")
    print()
    print(f"【形状评分】")
    print(f"  面积相似度: {area_dif:.2f}")
    print(f"  宽高比相似度: {wh_dif:.2f}")
    print(f"  形状综合得分: {shape_score:.2f}")
    print()
    print(f"【网格特征】")
    print(f"  原始余弦相似度: {grid_cosine_raw:.4f} (范围: -1 到 1)")
    print(f"  归一化后: {grid_similarity_normalized:.4f} (范围: 0 到 1)")
    print(f"  缩放到百分制: {grid_similarity_scaled:.2f} (范围: 0 到 100)")
    print(f"  平方后除以90: {(grid_similarity_scaled ** 2 / 90):.2f}")
    print(f"  最终网格相似度: {grid_similarity:.2f}")
    if grid_similarity >= 100:
        print(f"  ⚠️  注意: 结果被限制为100（可能因为相似度很高）")
    print()
    print(f"【骨架相似度】")
    print(f"  VGG特征相似度: {vgg_score:.2f}")
    print(f"  SSIM相似度: {ssim_score:.2f}")
    print(f"  Hausdorff距离得分: {hausdorff_dist_score:.2f}")
    print(f"  骨架综合得分: {skeleton_score:.2f}")
    print()
    print(f"【凸包相似度】")
    print(f"  凸包形状相似度: {hull_similarity_score:.2f}")
    print(f"  凸包距离得分: {distance_score:.2f}")
    print(f"  凸包综合得分: {hull_score:.2f}")
    print()
    print(f"【风格评分】")
    print(f"  风格得分: {style_scores:.2f} (已禁用)")
    print()
    
    base_score =gravity_score*0.5+shape_score*0.5
    if(base_score<60):
        print(f"【基础评分】: {base_score:.2f} (低于60分，直接返回基础分)")
        logging.info(f"basescore:{base_score}")
        print("=" * 60)
        return base_score
    else:
        score=base_score*0.15+grid_similarity*0.3+skeleton_score*0.25+hull_score*0.3
        print(f"【基础评分】: {base_score:.2f}")
        print(f"【最终评分】: {score:.2f}")
        print(f"  计算公式: base_score×0.15 + grid×0.3 + skeleton×0.25 + hull×0.3")
        print(f"  = {base_score:.2f}×0.15 + {grid_similarity:.2f}×0.3 + {skeleton_score:.2f}×0.25 + {hull_score:.2f}×0.3")
        print(f"  = {base_score*0.15:.2f} + {grid_similarity*0.3:.2f} + {skeleton_score*0.25:.2f} + {hull_score*0.3:.2f}")
        print(f"  = {score:.2f}")
        logging.info(f"score:{score}")
        print("=" * 60)
    return score

if __name__=="__main__":
    # 单张手写图片与目标图片路径
    source_image_path = r'C:\Users\24976\Desktop\code\text_score\single\16_c_r01_c01.jpg'  # 待评图片
    target_image_path = r'C:\Users\24976\Desktop\code\text_score\u5076.jpg'  # 目标模板

    logging.info(f"source: {source_image_path}")
    logging.info(f"target: {target_image_path}")
    
    print(f"\n正在对比图片...")
    print(f"手写图片: {os.path.basename(source_image_path)}")
    print(f"目标模板: {os.path.basename(target_image_path)}")
    
    score = main(source_image_path, target_image_path)
    
    print(f"\n最终结果:")
    print(f"图片: {os.path.basename(source_image_path)}")
    print(f"评分: {score:.2f} 分")