import cv2
import numpy as np
import os
import argparse

def ensure_dir(dir_path):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def crop_characters_from_image(image_path, output_dir, base_name=None):
    """
    从字帖图片中裁剪出所有单字
    
    参数:
        image_path: 输入图片路径
        output_dir: 输出文件夹
        base_name: 基础文件名（用于命名输出文件），如果为None则使用图片文件名
    
    返回:
        成功裁剪的字数，失败返回0
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return 0
    
    height, width = image.shape[:2]
    
    # 计算每个字的边长（正方形）
    char_size = width / 10.0
    
    # 计算12个字的总高度
    total_char_height = char_size * 12
    
    # 计算9个空隙的总高度
    total_gap_height = height - total_char_height
    
    # 计算每个空隙的高度
    gap_height = total_gap_height / 9.0
    
    print(f"图片尺寸: {width} x {height}")
    print(f"每个字边长: {char_size:.2f}")
    print(f"每个空隙高度: {gap_height:.2f}")
    
    # 如果没有基础名称，使用图片文件名（不含扩展名）
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    success_count = 0
    
    # 遍历12行10列
    for row in range(12):
        for col in range(10):
            # 计算当前字的位置
            # x坐标：列数 × 字边长
            x = int(col * char_size)
            
            # y坐标：行数 × (字边长 + 空隙高度) + 行数 × 空隙高度
            # 简化：行数 × 字边长 + 行数 × 空隙高度
            y = int(row * char_size + row * gap_height)
            
            # 裁剪区域
            x_end = int(x + char_size)
            y_end = int(y + char_size)
            
            # 确保不超出边界
            x_end = min(x_end, width)
            y_end = min(y_end, height)
            
            # 裁剪字符
            char_img = image[y:y_end, x:x_end]
            
            # 如果裁剪区域有效
            if char_img.size > 0:
                # 将图片resize到512×512
                char_img_resized = cv2.resize(char_img, (512, 512))
                
                # 保存文件，命名格式：原文件名_行_列.jpg（行和列从1开始）
                output_filename = f"{base_name}_r{row+1:02d}_c{col+1:02d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, char_img_resized)
                success_count += 1
    
    print(f"✓ 成功裁剪 {success_count} 个字")
    return success_count

def batch_crop_characters(input_dir, output_dir):
    """
    批量处理文件夹下的所有字帖图片
    
    参数:
        input_dir: 输入图片文件夹
        output_dir: 输出文件夹（所有单字保存在此文件夹下）
    """
    ensure_dir(output_dir)
    
    # 获取所有图像文件
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
    
    print(f"找到 {len(img_files)} 个图像文件，开始批量处理...\n")
    
    total_chars = 0
    success_files = 0
    fail_files = 0
    
    for idx, (fname, in_path) in enumerate(img_files, 1):
        print(f"[{idx}/{len(img_files)}] 处理: {fname}")
        base_name = os.path.splitext(fname)[0]
        
        char_count = crop_characters_from_image(in_path, output_dir, base_name)
        if char_count > 0:
            total_chars += char_count
            success_files += 1
        else:
            fail_files += 1
        print()
    
    print("=" * 50)
    print(f"批量处理完成！")
    print(f"  成功处理文件: {success_files} 个")
    print(f"  失败文件: {fail_files} 个")
    print(f"  总共裁剪字符: {total_chars} 个")
    print(f"  结果保存在: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从字帖图片中批量裁剪单字")
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹（所有单字保存在此）')
    parser.add_argument('--single', type=str, help='单张图片路径（用于测试）')
    parser.add_argument('--single-output', type=str, help='单张图片的输出文件夹')
    args = parser.parse_args()
    
    if args.single:
        # 单张图片处理
        output_dir = args.single_output if args.single_output else "output_chars"
        ensure_dir(output_dir)
        crop_characters_from_image(args.single, output_dir)
    elif args.input and args.output:
        # 批量处理
        batch_crop_characters(args.input, args.output)
    else:
        print("请指定输入和输出路径！")
        print("示例:")
        print("  批量处理: python crop_characters.py --input input_dir --output output_dir")
        print("  单张测试: python crop_characters.py --single image.jpg --single-output output_dir")

