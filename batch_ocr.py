import cv2
import os
import argparse
from paddleocr import PaddleOCR

def ocr_txt(img, ocr):
    """使用 PaddleOCR.predict() 识别文字（新版字典格式）"""
    result = ocr.predict(img)

    if not result or len(result) == 0:
        return None

    res = result[0]

    # 新版 PaddleOCR 识别文字在 rec_texts 中
    if "rec_texts" in res and len(res["rec_texts"]) > 0:
        return "".join(res["rec_texts"])

    return None


def split_text_to_chars(text):
    """将文字拆成单个字符"""
    if not text:
        return []
    return list(text)


def process_single_image(image_path, ocr):
    """处理单张图片"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ❌ 无法读取图片: {image_path}")
        return None

    # 🚀 缩放为 64×64
    img_small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    text = ocr_txt(img_small, ocr)
    return text  # 直接返回整段文字（可能是一个字或为空）


def batch_ocr(input_dir, output_file):
    print("正在初始化 OCR 模型...")
    ocr = PaddleOCR(lang='ch')
    print("OCR 模型加载完成！\n")

    # 获取所有图像文件
    img_files = []
    if os.path.isdir(input_dir):
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    img_files.append((filename, file_path))

    if not img_files:
        print(f"目录中未找到图片: {input_dir}")
        return

    # # ✔ 使用 **字符串排序**（严格字典序）
    # img_files.sort(key=lambda x: str(x[0]))

    print(f"找到 {len(img_files)} 张图片，开始识别...\n")

    success_count = 0
    fail_count = 0
    output_lines = []

    # 遍历每张图片
    for idx, (fname, img_path) in enumerate(img_files, 1):
        print(f"[{idx}/{len(img_files)}] 处理: {fname}")

        text = process_single_image(img_path, ocr)

        if text:
            chars = split_text_to_chars(text)
            print(f"  ✓ 识别到字符: {text}")
            success_count += 1
            # ✔ 每个字符单独一行
            for ch in chars:
                output_lines.append(f"{ch}    {fname}")
        else:
            print("  ⚠️ 未识别到文字")
            fail_count += 1
            # ✔ 在输出中保留一行标记
            output_lines.append(f"[未识别]    {fname}")

        print()

    # 保存结果
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print("=" * 50)
    print("OCR 识别完成！")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"输出行数: {len(output_lines)}")
    print(f"已保存到: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量 OCR 识别图片，每字一行输出")
    parser.add_argument("--input", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output", type=str, required=True, help="输出 txt 文件路径")
    args = parser.parse_args()

    batch_ocr(args.input, args.output)
