import cv2
import numpy as np
import os

def correct_ma(input_path, output_path, max_display_width=800):
    # 存储用户点击的4个点（原图坐标）
    points = []
    scale_ratio = 1.0  # 显示图与原图的缩放比例

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print("读取图像失败")
        return False

    h, w = img.shape[:2]
    # 缩小图像用于显示（不影响原图裁剪）
    if w > max_display_width:
        scale_ratio = max_display_width / w
        img_display = cv2.resize(img, (max_display_width, int(h * scale_ratio)))
    else:
        scale_ratio = 1.0
        img_display = img.copy()

    # 显示图像的副本（用于画点）
    img_copy = img_display.copy()

    # 对4个点排序（左上、左下、右下、右上）
    def order_points(pts):
        pts = np.array(pts, dtype="float32")
        xSorted = pts[np.argsort(pts[:, 0]), :]         # 按x排序
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]  # 左边两点按y排序
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]  # 右边两点按y排序
        (tr, br) = rightMost
        return np.array([tl, bl, br, tr], dtype="float32")

    # 利用4个点进行透视变换
    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, bl, br, tr) = rect

        # 计算变换后的宽高
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # 定义目标矩形的四个点
        dst = np.array([
            [0, 0],
            [0, maxHeight - 1],
            [maxWidth - 1, maxHeight - 1],
            [maxWidth - 1, 0]], dtype="float32")

        # 计算透视变换矩阵并裁剪
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    # 鼠标回调函数：左键点击记录点
    def mouse_handler(event, x, y, flags, param):
        nonlocal img_copy
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            # 将显示图坐标还原为原图坐标
            orig_x = int(x / scale_ratio)
            orig_y = int(y / scale_ratio)
            points.append((orig_x, orig_y))

            # 更新显示图（画上所有点）
            img_copy = img_display.copy()
            for p in points:
                disp_p = (int(p[0] * scale_ratio), int(p[1] * scale_ratio))
                cv2.circle(img_copy, disp_p, 5, (0, 255, 0), -1)
            cv2.imshow("image", img_copy)

    # 设置窗口和鼠标回调
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_handler)
    cv2.imshow("image", img_copy)

    print("请依次点击4个点。按 Delete 删除最后一个点，其他键确认并裁剪")
    while True:
        key = cv2.waitKey(0)
        if key in [8, 127]:  # Delete 键被按下
            if points:
                points.pop()  # 删除最后一个点
                img_copy = img_display.copy()
                for p in points:
                    disp_p = (int(p[0] * scale_ratio), int(p[1] * scale_ratio))
                    cv2.circle(img_copy, disp_p, 5, (0, 255, 0), -1)
                cv2.imshow("image", img_copy)
                print("已删除最后一个点")
        else:
            break  # 按下任意其他键，退出循环并进行裁剪

    cv2.destroyAllWindows()

    # 点不足4个，放弃裁剪
    if len(points) != 4:
        print("点数不足4个，取消裁剪")
        return False

    # 裁剪图像并保存到输出路径
    cropped = four_point_transform(img, points)
    cv2.imwrite(output_path, cropped)
    print(f"裁剪图已保存到: {output_path}")
    return True

def batch_process(input_dir, output_dir=None, suffix="_c", max_display_width=800):
    """
    批量处理目录中的所有图像文件
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则输出到输入目录
        suffix: 输出文件名后缀（在扩展名之前）
        max_display_width: 显示图像的最大宽度
    """
    # 支持的图像格式（小写，用于比较）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 获取所有图像文件（使用os.listdir避免Windows系统大小写不敏感导致的重复）
    image_files = []
    if os.path.isdir(input_dir):
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                # 将扩展名转为小写进行比较，避免大小写问题
                _, ext = os.path.splitext(filename.lower())
                if ext in image_extensions:
                    image_files.append(file_path)
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return
    
    # 对文件进行排序
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图像文件
    for idx, input_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {os.path.basename(input_path)}")
        
        # 生成输出文件名
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}{suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 如果输出文件已存在，询问是否跳过
        if os.path.exists(output_path):
            print(f"输出文件已存在: {output_path}")
            response = input("是否跳过此文件？(y/n，默认y): ").strip().lower()
            if response != 'n':
                print("跳过此文件")
                continue
        
        # 处理图像
        success = correct_ma(input_path, output_path, max_display_width)
        if success:
            print(f"✓ 成功处理: {os.path.basename(input_path)}")
        else:
            print(f"✗ 处理失败: {os.path.basename(input_path)}")
            response = input("是否继续处理下一个文件？(y/n，默认y): ").strip().lower()
            if response == 'n':
                print("用户取消批量处理")
                break
    
    print(f"\n批量处理完成！共处理 {idx} 个文件")

if __name__ == '__main__':
    import sys
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            # 如果是目录，批量处理
            output_dir = sys.argv[2] if len(sys.argv) > 2 else None
            batch_process(input_path, output_dir)
        else:
            # 如果是文件，单个处理
            output_path = sys.argv[2] if len(sys.argv) > 2 else None
            if output_path is None:
                output_path = os.path.join(
                    os.path.dirname(input_path), 
                    os.path.basename(input_path).replace(".jpg", "_c.jpg").replace(".png", "_c.png")
                )
            correct_ma(input_path, output_path)
    else:
        # 默认处理3500JPG目录
        input_dir = r"3500JPG"
        if os.path.exists(input_dir):
            print(f"批量处理目录: {input_dir}")
            batch_process(input_dir)
        else:
            # 如果3500JPG目录不存在，使用原来的单个文件处理方式
            input_path = r"image/test1.jpg"
            if os.path.exists(input_path):
                output_path = os.path.join(
                    os.path.dirname(input_path), 
                    os.path.basename(input_path).replace(".jpg", "_c.jpg")
                )
                correct_ma(input_path, output_path)
            else:
                print("请提供输入路径作为命令行参数，或确保存在默认目录/文件")
                print("用法: python correct_manual.py <输入路径> [输出路径/输出目录]")
