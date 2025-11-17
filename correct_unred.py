import cv2
import numpy as np
import os
import argparse

from PIL import Image, ImageEnhance

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def enhance_image(image):
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpen)
    return sharpened

def get_red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    return red_mask

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def shrink_polygon(corners, scale=0.95):
    """
    以中心为基准缩放四边形角点
    """
    center = np.mean(corners, axis=0)
    shrunken = (corners - center) * scale + center
    return shrunken

def detect_corners_from_mask_filtered(mask, save_dir, min_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ 未检测到任何轮廓")
        return None

    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not large_contours:
        print("⚠️ 没有符合面积阈值的轮廓")
        return None

    # 合并大轮廓并膨胀
    mask_temp = np.zeros_like(mask)
    cv2.drawContours(mask_temp, large_contours, -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_temp = cv2.dilate(mask_temp, kernel, iterations=1)

    contours2, _ = cv2.findContours(mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        print("⚠️ 膨胀后未检测到轮廓")
        return None

    max_contour = max(contours2, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)

    # 画凸包（仅在 save_dir 不为 None 时保存图片）
    contour_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, [hull], -1, (255, 0, 0), 3)
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, "convex_hull_filtered.jpg"), contour_img)

    # 多边形逼近
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2)
        return order_points(corners)

    print(f"⚠️ 逼近多边形顶点数是 {len(approx)}，使用minAreaRect替代")
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return order_points(box)

def warp_image(image, corners, save_dir):
    (tl, tr, br, bl) = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    cv2.imwrite(os.path.join(save_dir, "warped.jpg"), warped)
    return warped

def detect_grid_area(image_path, save_dir="output"):
    ensure_dir(save_dir)
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 图像加载失败")
        return

    enhanced = enhance_image(image)
    red_mask = get_red_mask(enhanced)
    cv2.imwrite(os.path.join(save_dir, "red_mask.jpg"), red_mask)

    corners = detect_corners_from_mask_filtered(red_mask, save_dir)
    if corners is None:
        print("❌ 无法检测到四角点")
        return

    # ✅ 缩放角点
    corners = shrink_polygon(corners, scale=0.99)

    vis = image.copy()
    for (x, y) in corners.astype(int):
        cv2.circle(vis, (x, y), 15, (0, 0, 255), -1)
    cv2.polylines(vis, [corners.astype(int)], isClosed=True, color=(0, 255, 255), thickness=5)
    cv2.imwrite(os.path.join(save_dir, "corners_marked.jpg"), vis)

    warp_image(image, corners, save_dir)
    print(f"✅ 田字格区域已提取并矫正，结果保存在 {save_dir}/warped.jpg")


def enhance(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 增强对比度和亮度
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(2)

    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(2)

    # 转回OpenCV格式
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image_cv


def remove_red_grid(image):
    """
    去除红色（包括深红、暗红、灰红）米字格线条 + 小像素残余
    返回去除红色网格后的彩色图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 扩展红色范围以适配更暗的红色
    lower_red1 = np.array([0, 10, 20])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 10, 20])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 膨胀操作放大红色区域
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    # 替换为白色
    result = image.copy()
    result[red_mask > 0] = [255, 255, 255]

    # 中值滤波去除孤立红点
    result = cv2.medianBlur(result, 3)
    return result


def denoise_select2(img):
    """
    去除红色米字格并二值化图像
    """
    enhanced_img = enhance(img)
    no_red_img = remove_red_grid(enhanced_img)
    gray = cv2.cvtColor(no_red_img, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)

    # 抗锯齿处理
    def anti_alias(binary_img):
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary_img, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated, (3, 3), 0)
        _, anti_aliased = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
        return anti_aliased

    binary = anti_alias(binary)
    return binary


def process_single_image(image, remove_grid_only=True):
    """
    处理单张图片：透视矫正 + 去除红色网格
    
    参数:
        image: 输入图像（BGR格式）
        remove_grid_only: 如果为True，只去除红色网格返回彩色图；如果为False，返回二值化图像
    
    返回:
        处理后的图像，如果检测失败返回None
    """
    # 增强锐化
    enhanced = enhance_image(image)
    # 获取红色区域mask（米字格）
    red_mask = get_red_mask(enhanced)
    # 透视矫正
    corners = detect_corners_from_mask_filtered(red_mask, save_dir=None)
    if corners is None:
        return None
    corners = shrink_polygon(corners, scale=0.99)
    (tl, tr, br, bl) = corners
    widthA = int(np.linalg.norm(br - bl))
    widthB = int(np.linalg.norm(tr - tl))
    maxWidth = max(widthA, widthB)
    heightA = int(np.linalg.norm(tr - br))
    heightB = int(np.linalg.norm(tl - bl))
    maxHeight = max(heightA, heightB)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    if remove_grid_only:
        # 只去除红色网格，返回彩色图像
        enhanced_warped = enhance(warped)
        result = remove_red_grid(enhanced_warped)
        return result
    else:
        # 二值化处理
        result = denoise_select2(warped)
        return result

def batch_process(input_dir, output_dir, remove_grid_only=True):
    """
    批量处理文件夹下所有图片，去除田字格并保存最终结果
    
    参数:
        input_dir: 输入图片文件夹
        output_dir: 输出结果文件夹（直接保存处理后的图片，不创建子文件夹）
        remove_grid_only: 如果为True，只去除红色网格保存彩色图；如果为False，保存二值化图像
    """
    ensure_dir(output_dir)
    
    # 获取所有图像文件（避免重复）
    img_files = []
    if os.path.isdir(input_dir):
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    img_files.append((filename, file_path))
    
    if not img_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return
    
    # 对文件进行排序
    img_files.sort(key=lambda x: x[0])
    
    print(f"找到 {len(img_files)} 个图像文件，开始批量处理...")
    
    success_count = 0
    fail_count = 0
    
    for idx, (fname, in_path) in enumerate(img_files, 1):
        print(f"\n[{idx}/{len(img_files)}] 处理: {fname}")
        
        # 读取原图
        image = cv2.imread(in_path)
        if image is None:
            print(f"  ❌ {fname} 加载失败，跳过")
            fail_count += 1
            continue
        
        # 处理图像
        result = process_single_image(image, remove_grid_only=remove_grid_only)
        if result is None:
            print(f"  ❌ {fname} 未能检测到田字格区域，跳过")
            fail_count += 1
            continue
        
        # 保存结果（保持原文件名和扩展名）
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, result)
        print(f"  ✓ 成功处理: {fname}")
        success_count += 1
    
    print(f"\n批量处理完成！")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {fail_count} 个")
    print(f"  结果保存在: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="田字格区域提取与去红色米字格批量处理工具")
    parser.add_argument('--input', type=str, help='输入图片文件夹')
    parser.add_argument('--output', type=str, help='输出结果文件夹')
    parser.add_argument('--single', type=str, help='单张图片路径')
    parser.add_argument('--single-output', type=str, help='单张图片输出路径')
    parser.add_argument('--binary', action='store_true', help='输出二值化图像（默认输出去除网格的彩色图像）')
    args = parser.parse_args()

    if args.input and args.output:
        # 批量处理
        batch_process(args.input, args.output, remove_grid_only=not args.binary)
    elif args.single:
        # 单张图片处理
        image = cv2.imread(args.single)
        if image is None:
            print(f"❌ {args.single} 图像加载失败")
        else:
            result = process_single_image(image, remove_grid_only=not args.binary)
            if result is None:
                print(f"❌ {args.single} 未能检测到田字格区域")
            else:
                output_path = args.single_output if args.single_output else "result.jpg"
                cv2.imwrite(output_path, result)
                print(f"✅ 处理完成，结果保存在: {output_path}")
    else:
        print("请通过 --input/--output 或 --single/--single-output 参数指定处理方式！")
        print("示例:")
        print("  批量处理: python correct_unred.py --input input_dir --output output_dir")
        print("  单张处理: python correct_unred.py --single image.jpg --single-output result.jpg")
        print("  输出二值化: python correct_unred.py --input input_dir --output output_dir --binary")
