import logging
import threading
import time
from pynput import keyboard

# 配置日志
logger = logging.getLogger("SimpleKeyboardHandler")
logger.setLevel(logging.DEBUG)

class SimpleKeyboardHandler:
    """简单的键盘快捷键处理器 - 处理全局键盘快捷键事件"""
    
    def __init__(self):
        self.listener = None
        self.running = False
        self.recording_callback = None
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.listener_thread = None
    
    def set_recording_callback(self, callback):
        """设置录音回调函数"""
        self.recording_callback = callback
    
    def on_press(self, key):
        """键盘按下事件处理"""
        try:
            # 检测Ctrl键
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = True
                logger.debug("Ctrl键已按下")
            # 检测Alt键
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_pressed = True
                logger.debug("Alt键已按下")
            # 检测R键
            elif hasattr(key, 'char') and key.char and key.char.lower() == 'r':
                logger.debug("检测到R键")
                # 检查是否同时按下了Ctrl和Alt
                if self.ctrl_pressed and self.alt_pressed:
                    logger.info("检测到Ctrl+Alt+R组合键")
                    if self.recording_callback:
                        # 直接执行回调（不创建新线程以简化）
                        self.recording_callback()
        except Exception as e:
            logger.error(f"处理按键事件时出错: {e}")
    
    def on_release(self, key):
        """键盘释放事件处理"""
        try:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = False
                logger.debug("Ctrl键已释放")
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_pressed = False
                logger.debug("Alt键已释放")
        except Exception as e:
            logger.error(f"处理释放事件时出错: {e}")
    
    def start(self):
        """启动键盘监听"""
        if self.running:
            logger.warning("键盘监听已经在运行中")
            return
        
        logger.debug("准备启动简单键盘监听器")
        self.running = True
        
        try:
            # 创建监听器
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
                suppress=False  # 不抑制按键事件
            )
            logger.debug("监听器对象创建成功")
            
            # 直接启动监听器（不使用额外的线程包装）
            self.listener.start()
            logger.info("简单键盘快捷键监听已启动 (Ctrl+Alt+R: 开始录音)")
        except Exception as e:
            logger.error(f"启动监听器失败: {e}")
            self.running = False
    
    def stop(self):
        """停止键盘监听"""
        if not self.running or self.listener is None:
            return
        
        try:
            self.running = False
            self.listener.stop()
            logger.info("简单键盘快捷键监听已停止")
        except Exception as e:
            logger.error(f"停止监听器失败: {e}")

# 创建单例实例
simple_keyboard_handler = SimpleKeyboardHandler()

# 导出便捷函数
def start_simple_keyboard_shortcuts(callback=None):
    """启动简单键盘快捷键监听"""
    if callback:
        simple_keyboard_handler.set_recording_callback(callback)
    simple_keyboard_handler.start()

def stop_simple_keyboard_shortcuts():
    """停止简单键盘快捷键监听"""
    simple_keyboard_handler.stop()

def set_simple_recording_callback(callback):
    """设置简单键盘快捷键的录音回调函数"""
    simple_keyboard_handler.set_recording_callback(callback)