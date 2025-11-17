import cv2
import numpy as np
import os

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

    # 画凸包
    contour_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, [hull], -1, (255, 0, 0), 3)
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

if __name__ == "__main__":
    detect_grid_area("test_red5.jpg", save_dir="output")
