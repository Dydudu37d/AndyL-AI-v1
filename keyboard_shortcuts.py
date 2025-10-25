import logging
import os
import platform
import sys
from pynput import keyboard
import threading
import stt

# 配置日志
logger = logging.getLogger("KeyboardShortcuts")

# 全局状态标志
is_keyboard_listener_available = True

# 检查操作系统和权限
if platform.system() == "Windows":
    try:
        # 尝试一种简单的权限检查方法
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if not is_admin:
            logger.warning("键盘监听可能需要管理员权限才能正常工作")
            print("⚠️  警告: 键盘监听可能需要管理员权限才能正常工作")
    except Exception as e:
        logger.error(f"检查管理员权限时出错: {e}")

class KeyboardShortcutsManager:
    """键盘快捷键管理器 - 处理全局键盘快捷键事件"""
    
    def __init__(self):
        self.listener = None
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.win_pressed = False  # 添加Win键状态变量
        self.recording_callback = None
        self.running = False
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
            # 检测Win键
            elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_r:
                self.win_pressed = True
                logger.debug("Win键已按下")
            # 检测R键的多种可能形式
            elif (hasattr(key, 'char') and key.char and key.char.lower() == 'r') or str(key) == '<82>' or str(key) == '\x12' or str(key) == 'r':
                logger.debug(f"检测到R键: {key}")
                # 检查是否同时按下了Ctrl和Alt
                if self.ctrl_pressed and self.alt_pressed:
                    logger.info("✅ 成功检测到Ctrl+Alt+R组合键")
                    if self.recording_callback:
                        # 在新线程中执行回调，避免阻塞
                        try:
                            threading.Thread(target=self.recording_callback).start()
                            logger.info("回调函数已在新线程中启动")
                        except Exception as callback_error:
                            logger.error(f"启动回调线程时出错: {callback_error}")
            # 额外的调试信息
            logger.debug(f"当前按键状态 - Ctrl: {self.ctrl_pressed}, Alt: {self.alt_pressed}, Win: {self.win_pressed}, Key: {key}")
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
            # 检测Win键释放
            elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_r:
                self.win_pressed = False
                logger.debug("Win键已释放")
            # 记录所有键的释放事件以便调试
            logger.debug(f"键已释放: {key}")
        except Exception as e:
            logger.error(f"处理释放事件时出错: {e}")
    
    def start(self):
        """启动键盘监听"""
        if self.running:
            logger.warning("键盘监听已经在运行中")
            return
        
        logger.debug("准备启动键盘监听器")
        self.running = True
        
        try:
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            logger.debug("监听器对象创建成功")
        except Exception as e:
            global is_keyboard_listener_available
            logger.error(f"创建监听器失败: {e}")
            is_keyboard_listener_available = False
            print(f"❌ 创建键盘监听器失败: {e}")
            print("💡 提示: 尝试以管理员权限运行程序")
            self.running = False
            return
        
        try:
            # 在新线程中启动监听器，避免阻塞主线程
            self.listener_thread = threading.Thread(target=self._run_listener)
            self.listener_thread.daemon = True
            logger.debug("监听器线程创建成功")
            self.listener_thread.start()
            logger.debug("监听器线程已启动")
        except Exception as e:
            logger.error(f"启动监听器线程失败: {e}")
            self.running = False
            return
        
        logger.info("键盘快捷键监听已启动 (Ctrl+Alt+R: 开始录音)")
        print("🎯 键盘快捷键监听已启动")
        print("   按下 Ctrl+Alt+R 可以快速切换语音模式")
        print("   注意: 在Windows系统上，可能需要管理员权限才能正常使用快捷键")
    
    def _run_listener(self):
        """在独立线程中运行监听器"""
        logger.debug("监听器线程开始执行")
        try:
            if self.listener:
                logger.debug("调用监听器的start方法")
                self.listener.start()
                logger.debug("监听器已启动，等待join")
                self.listener.join()
                logger.debug("监听器join完成")
        except Exception as e:
            global is_keyboard_listener_available
            logger.error(f"键盘监听器运行时出错: {e}")
            is_keyboard_listener_available = False
            print(f"❌ 键盘监听器运行出错: {e}")
            print("💡 提示: 如果快捷键不工作，请尝试以管理员权限运行程序")
        finally:
            self.running = False
            logger.debug("监听器线程执行完毕，running状态设为False")
    
    def stop(self):
        """停止键盘监听"""
        if not self.running or self.listener is None:
            return
        
        self.running = False
        self.listener.stop()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        
        logger.info("键盘快捷键监听已停止")

# 创建单例实例
keyboard_manager = KeyboardShortcutsManager()

# 导出便捷函数
def start_keyboard_shortcuts(callback=None):
    """启动键盘快捷键监听"""
    if callback:
        keyboard_manager.set_recording_callback(callback)
    keyboard_manager.start()
    

def stop_keyboard_shortcuts():
    """停止键盘快捷键监听"""
    keyboard_manager.stop()


def set_recording_callback(callback):
    """设置录音回调函数"""
    keyboard_manager.set_recording_callback(callback)


def get_keyboard_listener_status():
    """获取键盘监听器状态"""
    global is_keyboard_listener_available
    return {
        "available": is_keyboard_listener_available,
        "running": keyboard_manager.running,
        "platform": platform.system(),
        "needs_admin": platform.system() == "Windows"
    }