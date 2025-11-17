from PIL import Image, ImageDraw, ImageFont
import os

# 字符列表
characters = ["地", "围", "你", "草", "起","三"]

# 田字格背景图路径
grid_image_path = "/home/yly/workspace/text_score/data/image.png"  # 这里替换为田字格图片的路径

# 设置字体路径和字体大小
#fonts_folder = "/data/ylyu/text_data/ttf_tar"  # 这里替换为字体文件夹路径
fonts_folder = "/data/yly/text_data/ttf"
font_size = 700  # 字体大小

# 创建保存图像的文件夹
output_folder = "/data/ylyu/text_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载田字格图片
grid_image = Image.open(grid_image_path)

# 遍历每个字体文件
# 遍历字体文件夹中的所有 TTF 文件
i=1
for font_file in os.listdir(fonts_folder):

    if font_file.endswith((".ttf", ".otf")):  # 检查是否为 TTF 字体文件
        font_path = os.path.join(fonts_folder, font_file)

        font = ImageFont.truetype(font_path, font_size)

        # 遍历字符并生成图像
        
        for char in characters:
            # 为每个字符创建单独的文件夹
            char_folder = os.path.join(output_folder, char)
            if not os.path.exists(char_folder):
                os.makedirs(char_folder)

            # 创建田字格的副本，用于绘制字符
            image = grid_image.copy()
            draw = ImageDraw.Draw(image)

            
            
            # 获取字符的边界框（bbox）
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]  # 右边界减去左边界
            char_height = bbox[3] - bbox[1]  # 下边界减去上边界
            # 计算字符放置位置，确保字符在田字格内居中
            text_x = (image.width - char_width) // 2
            text_y = (image.height - char_height) // 2
            
            # 在图像上绘制字符
           
            draw.text((text_x, text_y), char, font=font, fill="black")

            # 保存图像到字符对应的文件夹
            image.save(os.path.join(char_folder, f"font{i}.png"))
            #image.save(os.path.join(char_folder, f"target.png"))
        i=i+1   

        print(f"已生成字体 {font_path.split('/')[-1]} 下的字符图像。")

print("所有字符图像已生成完毕！")
