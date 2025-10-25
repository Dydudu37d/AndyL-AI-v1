from PIL import Image, ImageDraw, ImageFont
import os

# 创建保存目录
if not os.path.exists('osu'):
    os.makedirs('osu')

# 创建黑色背景图像
image = Image.new('RGB', (400, 200), color='black')
draw = ImageDraw.Draw(image)

# 尝试使用不同的字体路径
try:
    # Windows默认字体路径
    font = ImageFont.truetype('arial.ttf', 80)
except:
    try:
        # Linux默认字体路径
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 80)
    except:
        # 使用默认字体
        font = ImageFont.load_default()

# 绘制白色数字
draw.text((50, 60), '7', font=font, fill='white')
draw.text((150, 60), '1', font=font, fill='white')
draw.text((250, 60), '9', font=font, fill='white')

# 保存图像
image_path = 'osu/test_digits_white_on_black.png'
image.save(image_path)
print(f"黑底白字测试图像已保存到: {image_path}")
print("图像包含数字: 7, 1, 9")
print("最小数字是: 1")
print("\n要测试此图像，请将其重命名为 osu/test_digits.png")