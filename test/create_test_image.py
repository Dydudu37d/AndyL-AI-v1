from PIL import Image, ImageDraw, ImageFont
import os

# 创建保存目录
if not os.path.exists('osu'):
    os.makedirs('osu')

# 创建白色背景图像
image = Image.new('RGB', (400, 200), color='white')
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

# 绘制数字
draw.text((50, 60), '8', font=font, fill='black')
draw.text((150, 60), '2', font=font, fill='black')
draw.text((250, 60), '5', font=font, fill='black')

# 保存图像
image_path = 'osu/test_digits.png'
image.save(image_path)
print(f"测试图像已保存到: {image_path}")
print("图像包含数字: 8, 2, 5")
print("最小数字是: 2")