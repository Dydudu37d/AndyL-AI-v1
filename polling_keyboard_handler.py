import logging
import threading
import time
import keyboard  # 这个库可能需要安装：pip install keyboard

# 配置日志
logger = logging.getLogger("PollingKeyboardHandler")
logger.setLevel(logging.DEBUG)

class PollingKeyboardHandler:
    """使用轮询方式的键盘快捷键处理器 - 避免事件监听问题"""
    
    def __init__(self):
        self.running = False
        self.recording_callback = None
        self.polling_thread = None
        self.polling_interval = 0.05  # 50ms检查一次状态
        
        # 防止重复触发的标志
        self.last_triggered = 0
        self.trigger_cooldown = 0.5  # 500ms的冷却时间
    
    def set_recording_callback(self, callback):
        """设置录音回调函数"""
        self.recording_callback = callback
    
    def _polling_loop(self):
        """按键状态轮询循环"""
        logger.debug("轮询线程已启动")
        
        while self.running:
            try:
                # 检查Ctrl+Alt+R组合键
                if (keyboard.is_pressed('ctrl') and 
                    keyboard.is_pressed('alt') and 
                    keyboard.is_pressed('r')):
                    
                    current_time = time.time()
                    # 检查是否在冷却期内
                    if current_time - self.last_triggered > self.trigger_cooldown:
                        logger.info("检测到Ctrl+Alt+R组合键")
                        self.last_triggered = current_time
                        
                        if self.recording_callback:
                            try:
                                self.recording_callback()
                            except Exception as e:
                                logger.error(f"执行回调函数时出错: {e}")
            except Exception as e:
                logger.error(f"轮询过程中出错: {e}")
            
            # 短暂休眠以避免CPU占用过高
            time.sleep(self.polling_interval)
        
        logger.debug("轮询线程已停止")
    
    def start(self):
        """启动键盘监听"""
        if self.running:
            logger.warning("键盘监听已经在运行中")
            return
        
        logger.debug("准备启动轮询键盘监听器")
        self.running = True
        
        try:
            # 创建并启动轮询线程
            self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
            self.polling_thread.start()
            logger.info("轮询键盘快捷键监听已启动 (Ctrl+Alt+R: 开始录音)")
            
        except Exception as e:
            logger.error(f"启动监听器失败: {e}")
            self.running = False
    
    def stop(self):
        """停止键盘监听"""
        if not self.running:
            return
        
        try:
            self.running = False
            
            if self.polling_thread:
                self.polling_thread.join(timeout=1.0)  # 等待线程结束，最多等待1秒
                logger.info("轮询键盘快捷键监听已停止")
        except Exception as e:
            logger.error(f"停止监听器失败: {e}")

# 创建单例实例
polling_keyboard_handler = PollingKeyboardHandler()

# 导出便捷函数
def start_polling_keyboard_shortcuts(callback=None):
    """启动轮询键盘快捷键监听"""
    if callback:
        polling_keyboard_handler.set_recording_callback(callback)
    polling_keyboard_handler.start()

def stop_polling_keyboard_shortcuts():
    """停止轮询键盘快捷键监听"""
    polling_keyboard_handler.stop()

def set_polling_recording_callback(callback):
    """设置轮询键盘快捷键的录音回调函数"""
    polling_keyboard_handler.set_recording_callback(callback)