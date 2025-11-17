#整张田字格
from tools import *
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import logging
from datetime import datetime
#from test_style import style_score
import config as c
from skimage.morphology import thin
from skimage.util import img_as_ubyte
from OCR import ocr_txt
from paddleocr import PaddleOCR,draw_ocr


def compute_score(source_image,target_image,idx,save_temp=False,save_log=False):
   

    bbox_s,hull_s = find_max_text_bbox(source_image)
    bbox_t,hull_t = find_max_text_bbox(target_image)

    #重心
    gravity_offset=gravity_offset_single(source_image)
    bounding_score=get_min_bounding_box_single(bbox_s,source_image)
    gravity_score=gravity_offset*c.gravity_weight1+bounding_score*c.gravity_weight2
    gravity_score = min(100, gravity_score ** 2 / 90)

    

    #面积得分
    cropped_s, area_s,wh_scales = crop_and_resize(source_image, bbox_s,target_size=(128,128))
    cropped_t, area_t,wh_scalet = crop_and_resize(target_image, bbox_t,target_size=(128,128))
   
    # print("area:",area_s,area_t)
    #ratio = min(area_s / area_t, area_t / area_s)
    ratio=abs(1-area_s/(256*256*0.75))
    area_dif = 100 * ratio
    #print(area_s)
    #area_dif=area_score(area_s)
    #print(area_dif)
    ratio = min(wh_scalet / wh_scales, wh_scales / wh_scalet)
    wh_dif = 100 * ratio
    shape_score=area_dif*c.area_weight+wh_dif*c.wh_weight

    #网格特征
    grid_similarity=gird_cosine_similarity(cropped_s,cropped_t,n=16)
    grid_similarity=normalized_cos(grid_similarity)*100
    grid_score = min(100, grid_similarity ** 2 / 90)

    #骨架相似度
    thin_image_s = thin(cropped_s)
    thin_image_s = img_as_ubyte(thin_image_s) 
    thin_image_t = thin(cropped_t)
    thin_image_t = img_as_ubyte(thin_image_t) 
    thin_image_s=cv2.bitwise_not(thin_image_s)
    thin_image_t=cv2.bitwise_not(thin_image_t)
    
    vgg_score=features_similarity(thin_image_s,thin_image_t) #vgg特征
    vgg_score=100-min(1, vgg_score ** 2 / 0.7)*100
    
    ssim_score, _ = ssim(thin_image_s, thin_image_t, full=True)
    ssim_score=min(1, ssim_score ** 2 / 0.55)*100
    
    pts1 = np.column_stack(np.where(thin_image_s == 255))  # 提取非零像素点位置
    pts2 = np.column_stack(np.where(thin_image_t == 255))
    
    hausdorff_dist = max(directed_hausdorff(pts1, pts2)[0], directed_hausdorff(pts2, pts1)[0])  # 计算Hausdorff距离
    normalized_dist = hausdorff_dist / np.sqrt(128**2 + 128**2)
    hausdorff_dist_score = (1 - normalized_dist) * 100
    hausdorff_dist_score = max(hausdorff_dist_score, 0) # 保证不为负数
    
    skeleton_score=vgg_score*c.ske_vgg_weight+ssim_score*c.ske_ssim_weight +hausdorff_dist_score* c.ske_hasdf_weight
    

    # 计算凸包相似度的评分
    hull_handwritten_norm=normalize_convex_hull(hull_s, target_size=(64,64))
    hull_template_norm=normalize_convex_hull(hull_t, target_size=(64,64))
    similarity = convex_hull_similarity(hull_template_norm, hull_handwritten_norm)
    hull_similarity_score = similarity * 100  # 相似度越高，评分越高

    distance = hausdorff_distance(hull_template_norm, hull_handwritten_norm)
    normalized_distance = distance / np.sqrt(64**2 + 64**2)  # 计算 Hausdorff 距离的评分
    distance_score = (1 - normalized_distance) * 100
    distance_score = max(distance_score, 0) # 保证不为负数
    
    hull_score=hull_similarity_score*c.hull_sim_weight+distance_score*c.hull_dis_weight
    # result1 = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(result1, [hull_s], -1, (0, 255, 0), 2)  # 在图像1上绘制凸包
    # cv2.imwrite("temp_images/hulls.png",result1)
    # result2 = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(result2, [hull_t], -1, (0, 255, 0), 2)  # 在图像2上绘制凸包
    if save_temp:
        pass
        cv2.imwrite(f"temp_images/{idx}_crops.png",cropped_s)
        cv2.imwrite(f"temp_images/{idx}_cropt.png",cropped_t)
        cv2.imwrite(f"temp_images/{idx}_thins.png",thin_image_s)
        cv2.imwrite(f"temp_images/{idx}_thint.png",thin_image_t)
    
    base_score =gravity_score*0.5+shape_score*0.5
    if(base_score<60):
        score=base_score
    else:
        score=base_score * c.base_weight +grid_similarity * c.grid_weight + skeleton_score * c.ske_weight +hull_score * c.hull_weight
        
    if save_log:
        logging.info(f"gravity {gravity_offset} {bounding_score} {gravity_score}")
        logging.info(f"area,wh {area_dif} {wh_dif} {shape_score}")
        logging.info(f"网格： {grid_score}")
        logging.info(f"alex： {vgg_score}")
        logging.info(f"SSIM相似度： {ssim_score}")
        logging.info(f"Hausdorff 距离： {hausdorff_dist_score}")
        logging.info(f"骨架得分：{skeleton_score}")
        logging.info(f"凸包： {hull_similarity_score} {distance_score} {hull_score}")
        #logging.info(f"风格：{style_scores}")
        logging.info(f"score:{score}")
    return score

if __name__=="__main__":
    
    image_path = c.image_path  # 田字格图片
    target_ttf = c.target_ttf  # 替换为target.png文件的路径
    rows = c.rows  # 例如10行
    cols = c.cols   # 例如8列
    resize_to = (256,256)  
    save_log=True
    if save_log:
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 创建一个以时间戳命名的日志文件
        log_filename = f"log/output_{timestamp}.log"
        # 配置日志
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.info(f"Image path: {image_path}")

    # 遍历文件夹中的所有图片（排除target.png）
    score_history=[]
    img = cv2.imread(image_path)
    #corrected_img=correct(img)
    cells=split_and_save_cells2(img, rows, cols, resize_to)
    ocr = PaddleOCR(use_gpu=False)
    txts=[]
    avg=[]
    
    for i in range(len(cells)):
    #for i in range(2):
        gray = cv2.cvtColor(cells[i], cv2.COLOR_BGR2GRAY)
        denoise_img=denoise_select_green(cells[i])

        enhance_img=enhance(cells[i])
        txt=ocr_txt(enhance_img,ocr)
        txts.append(txt)

        
        if c.save_temp:
            cv2.imwrite(f"temp_images/{i}_cell.png",cells[i])
            cv2.imwrite(f"temp_images/{i}_source.png",denoise_img)
        if txt: #文本有效
            
            if is_chinese(txt):
                logging.info(f"idx:{i}| txt:{txt} ---------------------------------")
                img_cv = render_text_to_cv(target_ttf, txt, font_size=120)
                # denoise_img=denoise_select1(cells[i]) #cells[i]是裁出来的单个田字格图像
                if c.save_temp:
                    
                    #cv2.imwrite(f"temp_images/{i}_target.png",img_cv)
                    pass
                score=compute_score(denoise_img,img_cv,i,False,save_log)   #  imgcv是黑底白字图像 256*256
                #logging.info(f" score:{score}")
                #score_history.append(score)
                score_history.append({"score": score, "txt": txt})  # 记录分数和文本

    # 计算平均分
    if score_history:
        mean_score = np.mean([entry["score"] for entry in score_history])
        logging.info(f"mean_score: {mean_score}, total_number: {len(score_history)}")
    else:
        mean_score = 0
        logging.warning("No valid scores recorded.")

    print(txts)
    #print(avg)
    # print(score_history)  # 输出完整的 score_history（包含分数和文本）
   