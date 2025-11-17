#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_font_image.py
升级版：安全文件名（uXXXX）、自动缩放、Pillow 保存（兼容 Windows Unicode 路径）
使用：
    单字符：
        python generate_font_image.py --ttf "myfont.ttf" --text 中 --output_dir out
    批量：
        python generate_font_image.py --ttf "myfont.ttf" --batch chars.txt --batch-output out
    交互：
        python generate_font_image.py --ttf "myfont.ttf"
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import math

# ---------------------------
# 配置（可改）
# ---------------------------
DEFAULT_IMG_SIZE = (512, 512)
DEFAULT_FONT_SIZE = 400
MIN_FONT_SIZE = 8
FONT_DECREASE_STEP = 2
JPEG_QUALITY = 95  # 保存 JPEG 质量


# ---------------------------
# 文件名策略：uXXXX（无反斜杠）
# 例如 '中' -> 'u4e2d'
# 多字符 -> 'u4f60u597d'
# ---------------------------
def to_u_code_filename(text: str) -> str:
    return "".join([f"u{ord(ch):04x}" for ch in text])


# ---------------------------
# 避免覆盖：如果存在同名文件，加后缀 _1 _2 ...
# ---------------------------
def ensure_path_unique(path: str) -> str:
    base, ext = os.path.splitext(path)
    candidate = path
    i = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate


# ---------------------------
# 计算文字 bbox（使用 mask 渲染以获得稳定 bbox）
# 返回 bbox 或 None
# ---------------------------
def get_text_bbox(font: ImageFont.FreeTypeFont, text: str, img_size):
    tmp = Image.new("L", img_size, 0)
    draw_tmp = ImageDraw.Draw(tmp)
    draw_tmp.text((0, 0), text, font=font, fill=255)
    return tmp.getbbox()


# ---------------------------
# 主渲染函数：自动缩放直到文字完全适配（或到最小字体）
# 使用 PIL 保存以避免 OpenCV 在 Windows 下 unicode 路径问题
# 返回保存的文件路径（如果保存），以及 PIL.Image 对象
# ---------------------------
def generate_font_image(ttf_path: str,
                        text: str,
                        output_path: str = None,
                        font_size: int = DEFAULT_FONT_SIZE,
                        img_size: tuple = DEFAULT_IMG_SIZE,
                        bg_color: tuple = (0, 0, 0),
                        text_color: tuple = (255, 255, 255),
                        auto_scale: bool = True):
    if not os.path.exists(ttf_path):
        raise FileNotFoundError(f"字体文件不存在: {ttf_path}")

    # clamp img_size
    img_w, img_h = int(img_size[0]), int(img_size[1])
    if img_w <= 0 or img_h <= 0:
        raise ValueError("图片尺寸必须为正整数")

    # 尝试加载字体（多次尝试不同大小）
    cur_size = int(font_size)
    last_success_font = None
    chosen_font = None
    chosen_bbox = None

    while cur_size >= MIN_FONT_SIZE:
        try:
            font = ImageFont.truetype(ttf_path, cur_size)
        except Exception as e:
            raise RuntimeError(f"无法加载字体: {e}")

        bbox = get_text_bbox(font, text, (img_w, img_h))
        if bbox is None:
            # 说明此字体在当前大小下未能渲染文本（可能字号过大导致 mask overflow）
            if not auto_scale:
                # 不缩放时直接抛错
                raise ValueError(f"无法渲染文字（bbox 为 None）：'{text}'")
            cur_size -= FONT_DECREASE_STEP
            continue

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 如果能完整放入（留 1 px margin），则接受
        if text_w <= img_w - 2 and text_h <= img_h - 2:
            chosen_font = font
            chosen_bbox = bbox
            break
        else:
            # 若允许自动缩放，继续减小字号
            if auto_scale:
                cur_size -= FONT_DECREASE_STEP
                continue
            else:
                # 不缩放则直接使用当前并让其裁剪（不推荐）
                chosen_font = font
                chosen_bbox = bbox
                break

    if chosen_font is None:
        # 最终仍没找到合适字号，用最小字号尝试一次
        chosen_font = ImageFont.truetype(ttf_path, max(MIN_FONT_SIZE, 10))
        chosen_bbox = get_text_bbox(chosen_font, text, (img_w, img_h))
        if chosen_bbox is None:
            raise ValueError(f"字体无法渲染该文字：'{text}' （即便使用最小字号）")

    # 计算居中坐标（基于 mask bbox）
    bbox = chosen_bbox
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img_w - text_w) // 2 - bbox[0]
    y = (img_h - text_h) // 2 - bbox[1]

    # 创建 RGBA（便于后续透明或其它操作），最后转换保存为 RGB
    pil_img = Image.new("RGBA", (img_w, img_h), (*bg_color, 255))
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=chosen_font, fill=(*text_color, 255))

    # 保存（如果给定路径），使用 Pillow 保存以避免 Windows 上 cv2 的路径问题
    saved_path = None
    if output_path:
        # 确保父目录存在
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # 归一化为绝对路径（利于诊断）
        output_path = os.path.abspath(output_path)
        output_path = ensure_path_unique(output_path)

        # 如果是 RGBA，要转换为 RGB（JPEG 不支持 alpha）
        save_img = pil_img.convert("RGB")
        # 使用 Pillow 的 save（quality 可配置）
        save_img.save(output_path, quality=JPEG_QUALITY)
        saved_path = output_path
        print(f"✓ 保存成功: {saved_path}")

    return saved_path, pil_img


# ---------------------------
# 批量接口（输入 txt 每行一个字符）
# ---------------------------
def batch_generate(input_file: str,
                   ttf_path: str,
                   output_dir: str,
                   font_size: int = DEFAULT_FONT_SIZE,
                   img_size: tuple = DEFAULT_IMG_SIZE,
                   bg_color: tuple = (0, 0, 0),
                   text_color: tuple = (255, 255, 255),
                   auto_scale: bool = True):
    if not os.path.exists(ttf_path):
        raise FileNotFoundError(f"字体文件不存在: {ttf_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n\r") for line in f]

    # 过滤空行并保持原顺序
    texts = [ln for ln in lines if ln and ln.strip()]

    print(f"找到 {len(texts)} 个非空行，开始生成（uXXXX 文件名）...\n")

    succ = 0
    fail = 0
    for idx, text in enumerate(texts, 1):
        # 每行可能包含多个字符；我们把整行作为一个字符串渲染（与你之前设计一致）
        if not text:
            print(f"[{idx}/{len(texts)}] 跳过空行")
            continue

        # 生成文件名（uXXXX...）
        uname = to_u_code_filename(text)
        filename = f"{uname}.jpg"
        out_path = os.path.join(output_dir, filename)

        print(f"[{idx}/{len(texts)}] 生成: '{text}' -> {filename}")

        try:
            saved, _ = generate_font_image(
                ttf_path=ttf_path,
                text=text,
                output_path=out_path,
                font_size=font_size,
                img_size=img_size,
                bg_color=bg_color,
                text_color=text_color,
                auto_scale=auto_scale
            )
            succ += 1
        except Exception as e:
            fail += 1
            print(f"  ❌ 生成失败: {e}")

    print("\n批量任务完成")
    print(f"成功: {succ} 失败: {fail}")


# ---------------------------
# 单字符/多字符模式与交互
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="字体图片生成工具（uXXXX 文件名、安全、自动缩放）")
    parser.add_argument("--ttf", type=str, required=True, help="TTF 字体文件路径")
    parser.add_argument("--text", type=str, help="要生成的文字（单字符或多字符）")
    parser.add_argument("--output-dir", type=str, help="输出目录（单字符模式）")
    parser.add_argument("--output", type=str, help="输出完整文件路径（覆盖 output-dir）")
    parser.add_argument("--batch", type=str, help="批量模式：每行一个字符/字符串的文本文件（UTF-8）")
    parser.add_argument("--batch-output", type=str, help="批量输出目录")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE, help="起始字体大小")
    parser.add_argument("--size", type=int, nargs=2, default=list(DEFAULT_IMG_SIZE), metavar=("W", "H"),
                        help="图片尺寸，示例: --size 512 512")
    parser.add_argument("--bg-color", type=int, nargs=3, default=[255, 255, 255], metavar=("R", "G", "B"),
                        help="背景颜色")
    parser.add_argument("--text-color", type=int, nargs=3, default=[0, 0, 0], metavar=("R", "G", "B"),
                        help="文字颜色")
    parser.add_argument("--no-auto-scale", action="store_true", help="禁用自动缩放（使用固定 font-size）")

    args = parser.parse_args()

    img_size = (int(args.size[0]), int(args.size[1]))
    bg_color = tuple(int(x) for x in args.bg_color)
    text_color = tuple(int(x) for x in args.text_color)
    auto_scale = not args.no_auto_scale

    if args.batch:
        out_dir = args.batch_output if args.batch_output else "output_fonts"
        batch_generate(
            input_file=args.batch,
            ttf_path=args.ttf,
            output_dir=out_dir,
            font_size=args.font_size,
            img_size=img_size,
            bg_color=bg_color,
            text_color=text_color,
            auto_scale=auto_scale
        )
        return

    # 单字符/多字符模式或交互
    if args.text:
        # 输出路径优先使用 --output（完整路径），否则使用 --output-dir 或当前目录
        if args.output:
            out_path = args.output
        else:
            out_dir = args.output_dir if args.output_dir else "."
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            fname = f"{to_u_code_filename(args.text)}.jpg"
            out_path = os.path.join(out_dir, fname)

        out_path = ensure_path_unique(out_path)
        generate_font_image(
            ttf_path=args.ttf,
            text=args.text,
            output_path=out_path,
            font_size=args.font_size,
            img_size=img_size,
            bg_color=bg_color,
            text_color=text_color,
            auto_scale=auto_scale
        )
        return

    # 交互模式
    print("进入交互模式。输入 exit 或 quit 退出。")
    while True:
        try:
            text = input("文字: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            print("退出。")
            break

        fname = f"{to_u_code_filename(text)}.jpg"
        out_dir = "."
        out_path = os.path.join(out_dir, fname)
        out_path = ensure_path_unique(out_path)

        try:
            generate_font_image(
                ttf_path=args.ttf,
                text=text,
                output_path=out_path,
                font_size=args.font_size,
                img_size=img_size,
                bg_color=bg_color,
                text_color=text_color,
                auto_scale=auto_scale
            )
        except Exception as e:
            print(f"生成失败: {e}")


if __name__ == "__main__":
    main()
