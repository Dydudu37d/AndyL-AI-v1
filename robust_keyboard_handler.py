import logging
import threading
import time
from pynput import keyboard

# 配置日志
logger = logging.getLogger("RobustKeyboardHandler")
logger.setLevel(logging.DEBUG)

class RobustKeyboardHandler:
    """健壮的键盘快捷键处理器 - 改进的组合键检测逻辑"""
    
    def __init__(self):
        self.listener = None
        self.running = False
        self.recording_callback = None
        
        # 按键状态跟踪
        self.key_states = {
            'ctrl': False,
            'alt': False,
            'r': False
        }
        
        # 状态检查定时器
        self.check_timer = None
        self.check_interval = 0.05  # 50ms检查一次状态
        
        # 防止重复触发的标志
        self.last_triggered = 0
        self.trigger_cooldown = 0.5  # 500ms的冷却时间
    
    def set_recording_callback(self, callback):
        """设置录音回调函数"""
        self.recording_callback = callback
    
    def on_press(self, key):
        """键盘按下事件处理"""
        try:
            # 更新按键状态
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.key_states['ctrl'] = True
                logger.debug("Ctrl键已按下")
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.key_states['alt'] = True
                logger.debug("Alt键已按下")
            elif hasattr(key, 'char') and key.char and key.char.lower() == 'r':
                self.key_states['r'] = True
                logger.debug("R键已按下")
            
            # 检查组合键
            self._check_combination()
        except Exception as e:
            logger.error(f"处理按键事件时出错: {e}")
    
    def on_release(self, key):
        """键盘释放事件处理"""
        try:
            # 更新按键状态
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.key_states['ctrl'] = False
                logger.debug("Ctrl键已释放")
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.key_states['alt'] = False
                logger.debug("Alt键已释放")
            elif hasattr(key, 'char') and key.char and key.char.lower() == 'r':
                self.key_states['r'] = False
                logger.debug("R键已释放")
        except Exception as e:
            logger.error(f"处理释放事件时出错: {e}")
    
    def _check_combination(self):
        """检查是否满足组合键条件"""
        current_time = time.time()
        
        # 检查是否所有键都被按下且不在冷却期
        if (self.key_states['ctrl'] and 
            self.key_states['alt'] and 
            self.key_states['r'] and 
            current_time - self.last_triggered > self.trigger_cooldown):
            
            logger.info("检测到Ctrl+Alt+R组合键")
            self.last_triggered = current_time
            
            if self.recording_callback:
                try:
                    self.recording_callback()
                except Exception as e:
                    logger.error(f"执行回调函数时出错: {e}")
    
    def _start_check_timer(self):
        """启动状态检查定时器"""
        if self.check_timer and self.check_timer.is_alive():
            return
        
        def check_loop():
            while self.running:
                self._check_combination()
                time.sleep(self.check_interval)
        
        self.check_timer = threading.Thread(target=check_loop, daemon=True)
        self.check_timer.start()
    
    def start(self):
        """启动键盘监听"""
        if self.running:
            logger.warning("键盘监听已经在运行中")
            return
        
        logger.debug("准备启动健壮键盘监听器")
        self.running = True
        
        try:
            # 重置按键状态
            self.key_states = {'ctrl': False, 'alt': False, 'r': False}
            
            # 创建监听器
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
                suppress=False
            )
            logger.debug("监听器对象创建成功")
            
            # 启动监听器
            self.listener.start()
            logger.info("健壮键盘快捷键监听已启动 (Ctrl+Alt+R: 开始录音)")
            
            # 启动状态检查定时器
            self._start_check_timer()
            
        except Exception as e:
            logger.error(f"启动监听器失败: {e}")
            self.running = False
    
    def stop(self):
        """停止键盘监听"""
        if not self.running:
            return
        
        try:
            self.running = False
            
            if self.listener:
                self.listener.stop()
                logger.info("健壮键盘快捷键监听已停止")
        except Exception as e:
            logger.error(f"停止监听器失败: {e}")

# 创建单例实例
robust_keyboard_handler = RobustKeyboardHandler()

# 导出便捷函数
def start_robust_keyboard_shortcuts(callback=None):
    """启动健壮键盘快捷键监听"""
    if callback:
        robust_keyboard_handler.set_recording_callback(callback)
    robust_keyboard_handler.start()

def stop_robust_keyboard_shortcuts():
    """停止健壮键盘快捷键监听"""
    robust_keyboard_handler.stop()

def set_robust_recording_callback(callback):
    """设置健壮键盘快捷键的录音回调函数"""
    robust_keyboard_handler.set_recording_callback(callback)