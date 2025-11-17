import cv2
import os
import numpy as np

def split_and_save_cells2(img, rows, cols, resize_to,save_is=False):
    """
    绿色格子 存在空隙
    """
    
    # 读取图像
    
    h, w = img.shape[:2]
    
    cell_width = w // cols
    cell_height = cell_width

    # 如果保存目录不存在就创建
    if save_is:

        save_dir = '/home/yly/workspace/text_score/testimg/cells'  # 保存目录
        os.makedirs(save_dir, exist_ok=True)

    idx = 0
    cells=[]
    row_gap=(h-cell_height*4)//3
    for r in range(rows):
        for c in range(cols):
            # 处理最后一行和最后一列，不丢像素
            x1 = c * cell_width
            y1 = r * cell_height+r*row_gap
            x2 = (c + 1) * cell_width if c != cols - 1 else w
            y2 = y1+cell_height if r != rows - 1 else h

            # 裁剪小格子
            cell = img[y1:y2, x1:x2]

            # Resize到指定大小
            shrink_size = cell_width//20  # 每边收缩的像素数
            cell_shrunk = cell[shrink_size:-shrink_size, shrink_size:-shrink_size]
            cell_resized = cv2.resize(cell_shrunk, resize_to)
            cells.append(cell_resized)


            # # 保存
            if save_is:
                save_path = os.path.join(save_dir, f"cell_{idx:03d}.png")
                cv2.imwrite(save_path, cell_resized)
            idx += 1

    #print(f"共 {idx} 个小格子")
    return cells
def split_and_save_cells1(img, rows, cols, resize_to,save_is=False):
    """
    米字格  紧密分布
    """
    # 读取图像
    
    h, w = img.shape[:2]
    cell_height = h // rows
    cell_width = w // cols

    # 如果保存目录不存在就创建
    if save_is:

        save_dir = '/home/yly/workspace/text_score/testimg/cells'  # 保存目录
        os.makedirs(save_dir, exist_ok=True)

    idx = 0
    cells=[]
    for r in range(rows):
        for c in range(cols):
            # 处理最后一行和最后一列，不丢像素
            x1 = c * cell_width
            y1 = r * cell_height
            x2 = (c + 1) * cell_width if c != cols - 1 else w
            y2 = (r + 1) * cell_height if r != rows - 1 else h

            # 裁剪小格子
            cell = img[y1:y2, x1:x2]

            # Resize到指定大小
            cell_resized = cv2.resize(cell, resize_to)
            cells.append(cell_resized)


            # # 保存
            if save_is:
                save_path = os.path.join(save_dir, f"cell_{idx:03d}.png")
                cv2.imwrite(save_path, cell_resized)
            idx += 1

    #print(f"共 {idx} 个小格子")
    return cells

# ===== 使用示例 =====
if __name__=='__main__':
    img_path = '/home/yly/workspace/text_score/testimg/corrected_img2.png'
    img = cv2.imread(img_path)
    
    rows = 4  # 例如10行
    cols = 8   # 例如8列
    resize_to = (128,128)  # 统一resize成256*256像素

    cells=split_and_save_cells2(img, rows, cols, resize_to)
    for i in range(len(cells)):
        cv2.imwrite(f"/home/yly/workspace/text_score/testimg/cells/{i}_cell.png",cells[i])


