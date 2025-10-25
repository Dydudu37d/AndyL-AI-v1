from PIL import ImageGrab
import os
import datetime

def take_full_screenshot():
    """
    截取整个屏幕并保存到screenshots目录
    """
    # 创建screenshots目录（如果不存在）
    screenshots_dir = os.path.join(os.getcwd(), 'screenshots')
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(screenshots_dir, filename)
    
    try:
        # 截取整个屏幕
        print("正在截取整个屏幕...")
        screenshot = ImageGrab.grab()
        
        # 保存截图
        screenshot.save(filepath)
        print(f"截图已保存: {filepath}")
        
        return filepath
    except Exception as e:
        print(f"截图过程中发生错误: {e}")
        return None

if __name__ == "__main__":
    take_full_screenshot()