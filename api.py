from tools import *
from main_m import compute_score
import numpy as np
from paddleocr import PaddleOCR,draw_ocr



target_ttfs = {
    1: '/data/yly/text_data/ttf_tar/华栋正楷第三版 Regular.ttf', 
    2: '/data/yly/text_data/ttf_tar/另外一个字体.ttf',           
    3: '/data/yly/text_data/ttf_tar/第三个字体.ttf',               
}
def ocr_txt(img,ocr):
    result = ocr.ocr(img,rec=True)
    result = result[0]
    if result==None:
        return None
    txts=result[0][1][0]
    return txts
def apifun(paper,score_select,ttf_select,img):
    ocr = PaddleOCR(use_gpu=False) # need to run only once to download and load model into memory

    target_ttf = target_ttfs[ttf_select]  # 替换为target.png文件的路径
    if(paper==1):
        #绿色4行8列有空隙
        resize_to = (512,512)  # 统一resize成256*256像素
        # 遍历文件夹中的所有图片（排除target.png）
        score_history=[]
        #corrected_img=correct(img)
        #img = cv2.imread(img)
        cells=split_and_save_cells2(img, 4, 8, resize_to)
        txts=[]
        for i in range(len(cells)):
            txt=ocr_txt(cells[i],ocr)
            denoise_img=denoise_select_green(cells[i])
            txts.append(txt)
            if txt: #文本有效
                if is_chinese(txt):
                    img_cv = render_text_to_cv(target_ttf, txt, font_size=90) 
                    score=compute_score(denoise_img,img_cv,i)   #128 *128  imgcv是黑底白字图像 128*128
                    score_history.append({"score": score, "txt": txt})  # 记录分数和文本
    elif(paper==2):
        #红色田字格
        resize_to = (512,512)  # 统一resize成256*256像素
        # 遍历文件夹中的所有图片（排除target.png）
        score_history=[]
        #corrected_img=correct(img)
        #img = cv2.imread(img)
        cells=split_and_save_cells1(img, 8, 11, resize_to)
        txts=[]
        for i in range(len(cells)):
            txt=ocr_txt(cells[i],ocr)
            denoise_img=denoise_select_red(cells[i])
            txts.append(txt)
            if txt: #文本有效
                if is_chinese(txt):
                    img_cv = render_text_to_cv(target_ttf, txt, font_size=100) 
                    score=compute_score(denoise_img,img_cv,i)   #128 *128  imgcv是黑底白字图像 128*128
                    score_history.append({"score": score, "txt": txt})  # 记录分数和文本
        pass
    else:
        pass

    # 计算平均分
    if score_history:
        mean_score = np.mean([entry["score"] for entry in score_history])
    else:
        mean_score = 0

    if score_select==1:
        return mean_score
    elif score_select==2:
        return score_history