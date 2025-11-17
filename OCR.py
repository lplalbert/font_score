from tools.ttftoimg import render_text_to_cv
import cv2
from paddleocr import PaddleOCR,draw_ocr





def ocr_txt(img,ocr):
    
    result = ocr.ocr(img,rec=True)
    result = result[0]
    #print("result",result)
    if result==None:
        return None
    txts=result[0][1][0]
    #txts=result
    return txts

if __name__=="__main__":
    
    ocr = PaddleOCR(use_gpu=False) # need to run only once to download and load model into memory
    img_path = '/home/yly/workspace/text_score/image/test1_c.jpg'
    img = cv2.imread(img_path)
    result=ocr_txt(img,ocr)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
            
    #         print(line)

    # draw result
    from PIL import Image
    print("result",result)
    if result:
        print("sss")
    # result = result[0]
    # # print("result[0]",result)
    # # print("result[0][0]",result[0])
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # print("box:",boxes)
    # txts = [line[1][0] for line in result]
    # print("txt",txts)
    # for line in result:
    #     print("line:",line)
    # boxes=result[0][0]
    # txts=result[0][1][0]
    # print("box:",boxes)
    # print("txt",txts)
    # ttf_path = '/data/yly/text_data/ttf_tar/华栋正楷第三版 Regular.ttf'
    # font_size = 128
    # text = '起'
    # img_cv = render_text_to_cv(ttf_path, txts, font_size)

    # # 也可以直接用cv2.imwrite保存
    # cv2.imwrite('/home/yly/workspace/text_score/testimg/cat1.png', img_cv)

    # im_show = draw_ocr(image, boxes, txts=None, scores=None, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)

    # im_show.save('/home/yly/workspace/text_score/testimg/resultimg.jpg')