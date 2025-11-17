def area_score(area_of_text):
    # 正方形的面积
    square_area = 128*128
    
    # 计算字的面积占正方形面积的百分比
    text_area_ratio = (area_of_text / square_area) * 100
    #print(text_area_ratio)
    
    # 如果字的面积小于 25%，分数为 0
    if text_area_ratio < 50:
        # 超出部分的比例
        excess_ratio = (50-text_area_ratio)
        # 从 100 分中扣除
        score = 100 - excess_ratio
        
    
    # 如果字的面积大于 25%，扣除超出部分的比例
    else :
        # 超出部分的比例
        excess_ratio = (text_area_ratio - 50)
        # 从 100 分中扣除
        score = 100 - excess_ratio
    #print(score)
    return max(0, score)  # 确保分数不会小于 0
        
    
   