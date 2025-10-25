import cv2
import numpy as np
import pyautogui
import time
from PIL import ImageGrab
import threading
import logging
import cv2

# 配置日志
logger = logging.getLogger("ScreenCapture")
logger.setLevel(logging.INFO)

class ScreenCapture:
    def __init__(self, region=None, width=84, height=84, capture_interval=0.7):
        # region格式: (x, y, width, height)
        self.region = region
        self.width = width
        self.height = height
        self.capture_interval = capture_interval  # 截图间隔(秒)
        
        # 控制标志
        self.is_capturing = False
        self.capture_thread = None
        
        # 回调函数
        self.screenshot_callback = None
        
        # 最近的截图
        self.latest_screenshot = None
        self.latest_observation = None
        
        # 鼠标控制器引用
        self.mouse_controller = None
        
        # AI大脑引用
        self.ai_brain = None
    
    def set_mouse_controller(self, mouse_controller):
        """设置鼠标控制器"""
        self.mouse_controller = mouse_controller
    
    def set_ai_brain(self, ai_brain):
        """设置AI大脑"""
        self.ai_brain = ai_brain
    
    def capture_screen(self):
        """捕获屏幕图像"""
        try:
            if self.region:
                screenshot = ImageGrab.grab(bbox=self.region)
            else:
                screenshot = ImageGrab.grab()
            
            # 转换为OpenCV格式
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.latest_screenshot = frame
            return frame
        except Exception as e:
            logger.error(f"捕获屏幕失败: {e}")
            return None
    
    def save_screenshot(self, file_path=None, screenshot=None):
        """保存截图到文件
        
        参数:
            file_path: 保存路径，如果为None则自动生成
            screenshot: 要保存的截图，如果为None则使用最近的截图
        
        返回:
            保存成功返回True，失败返回False
        """
        try:
            # 使用提供的截图或最近的截图
            if screenshot is None:
                screenshot = self.latest_screenshot
                
                # 如果最近的截图也为None，则捕获新的截图
                if screenshot is None:
                    screenshot = self.capture_screen()
                    if screenshot is None:
                        logger.error("没有可保存的截图")
                        return False
            
            # 如果未提供保存路径，则自动生成
            if file_path is None:
                # 生成包含时间戳的文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = f"screenshots/screenshot_{timestamp}.png"
                
                # 确保screenshots目录存在
                import os
                os.makedirs("screenshots", exist_ok=True)
            
            # 保存截图
            cv2.imwrite(file_path, screenshot)
            logger.info(f"截图已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
            return False
        
    def preprocess_image(self, image):
        """预处理图像以供神经网络使用"""
        if image is None:
            return None
        
        # 调整大小
        resized = cv2.resize(image, (self.width, self.height))
        
        # 转换为RGB并归一化
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # 添加批次维度
        processed = np.expand_dims(normalized, axis=0)
        
        self.latest_observation = processed
        return processed
        
    def get_observation(self):
        """获取当前屏幕的处理后的观察值"""
        screen = self.capture_screen()
        if screen is None:
            return self.latest_observation
        
        observation = self.preprocess_image(screen)
        return observation
        
    def show_captured_screen(self, duration=2):
        """显示捕获的屏幕图像，用于调试"""
        screen = self.capture_screen()
        if screen is not None:
            cv2.imshow("Captured Screen", screen)
            cv2.waitKey(int(duration * 1000))
            cv2.destroyAllWindows()
        
    def benchmark_capture(self, iterations=100):
        """测试屏幕捕获性能"""
        total_time = 0
        
        for i in range(iterations):
            start_time = time.time()
            self.capture_screen()
            end_time = time.time()
            total_time += (end_time - start_time)
            
        avg_time = total_time / iterations
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"平均捕获时间: {avg_time:.4f}秒")
        print(f"帧率: {fps:.2f} FPS")
        
        return fps
        
    def set_screenshot_callback(self, callback):
        """设置截图回调函数"""
        self.screenshot_callback = callback
        
    def _capture_loop(self):
        """截图循环"""
        while self.is_capturing:
            start_time = time.time()
            
            # 捕获屏幕
            screenshot = self.capture_screen()
            
            # 处理截图并执行控制逻辑
            if screenshot is not None:
                self._process_screenshot_and_control(screenshot)
                
                # 调用回调函数
                if self.save_path:
                    self.save_screenshot(self.save_path, screenshot)
                if self.screenshot_callback:
                    try:
                        self.screenshot_callback(screenshot)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")
            
            # 计算需要等待的时间以保持截图间隔
            elapsed_time = time.time() - start_time
            wait_time = max(0, self.capture_interval - elapsed_time)
            time.sleep(wait_time)
        
    def _process_screenshot_and_control(self, screenshot):
        """处理截图并执行电脑控制"""
        try:
            # 如果有AI大脑，可以让AI分析截图并生成控制指令
            if self.ai_brain and self.mouse_controller:
                # 这里可以实现更复杂的屏幕分析逻辑
                # 简化版本：假设我们只是根据简单规则进行控制
                self._simple_control_logic()
        except Exception as e:
            logger.error(f"处理截图和控制失败: {e}")
            
    def _simple_control_logic(self):
        """简单的控制逻辑示例"""
        # 这里可以根据实际需求实现复杂的控制逻辑
        # 示例：随机移动鼠标（实际应用中应该基于AI分析或具体需求）
        # 当前不实现具体逻辑，需要根据实际需求定制
        pass
        
    def start_capturing(self,save_path:str = None):
        """开始定时截图"""
        if not self.is_capturing:
            self.save_path = save_path
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            logger.info(f"开始定时截图，间隔: {self.capture_interval}秒")
        
    def stop_capturing(self):
        """停止定时截图"""
        if self.is_capturing:
            self.is_capturing = False
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
            logger.info("停止定时截图")

# 为了与main.py集成，提供一个简单的演示函数
def demo_screen_control(mouse_controller, ai_brain=None):
    """演示屏幕控制功能"""
    # 创建ScreenCapture实例
    screen_capture = ScreenCapture(capture_interval=0.7)
    
    # 设置控制器和AI大脑
    screen_capture.set_mouse_controller(mouse_controller)
    if ai_brain:
        screen_capture.set_ai_brain(ai_brain)
    
    # 定义截图回调函数，用于显示截图信息
    def on_screenshot(screenshot):
        height, width = screenshot.shape[:2]
        logger.info(f"捕获到屏幕截图: {width}x{height}")
    
    # 设置回调
    screen_capture.set_screenshot_callback(on_screenshot)
    
    # 开始捕获
    screen_capture.start_capturing()
    
    return screen_capture

if __name__ == "__main__":
    # 测试屏幕捕获
    capture = ScreenCapture(width=84, height=84, capture_interval=0.7)
    print("测试屏幕捕获...")
    capture.show_captured_screen()
    print("测试性能...")
    capture.benchmark_capture()
    
    print("开始定时截图测试(5秒)...")
    capture.start_capturing()
    time.sleep(5)
    capture.stop_capturing()
    print("定时截图测试结束")