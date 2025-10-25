import logging
import time
import threading
from pynput import keyboard
from keyboard_shortcuts import KeyboardShortcutsManager

# 配置日志为DEBUG级别，以便查看更详细的信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugKeyboardShortcuts")

# 创建一个管理器实例
keyboard_manager = KeyboardShortcutsManager()

# 模拟录音状态
is_recording = False

# 模拟录音回调函数
def mock_recording_callback():
    global is_recording
    is_recording = not is_recording
    status = "开始录音" if is_recording else "停止录音"
    print(f"\n🎤 {status}! (通过Ctrl+Alt+R触发)")
    logger.info(f"录音状态已切换为: {status}")

# 添加额外的调试日志
def debug_on_press(key):
    try:
        # 打印按下的键
        if hasattr(key, 'char'):
            logger.debug(f"按下按键: {key.char}")
        else:
            logger.debug(f"按下特殊键: {key}")
    except Exception as e:
        logger.error(f"处理调试按键事件时出错: {e}")

def debug_on_release(key):
    try:
        # 打印释放的键
        if hasattr(key, 'char'):
            logger.debug(f"释放按键: {key.char}")
        else:
            logger.debug(f"释放特殊键: {key}")
    except Exception as e:
        logger.error(f"处理调试释放事件时出错: {e}")

def debug_keyboard_shortcuts():
    """详细调试键盘快捷键功能"""
    print("======= 键盘快捷键详细调试 =======")
    print("此测试将显示所有按键事件和状态变化")
    print("请按Ctrl+Alt+R组合键来测试")
    print("按Esc键退出测试")
    print("============================")
    
    # 设置回调函数
    keyboard_manager.set_recording_callback(mock_recording_callback)
    
    # 创建一个额外的监听器来记录所有按键事件
    debug_listener = keyboard.Listener(
        on_press=debug_on_press,
        on_release=debug_on_release
    )
    
    # 启动调试监听器
    debug_listener.start()
    
    # 启动键盘快捷键监听器
    keyboard_manager.start()
    
    try:
        # 显示当前状态
        print("\n监听中...")
        while True:
            # 检查是否按下Esc键退出
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x1b':  # Esc键
                    break
            
            # 显示当前组合键状态
            time.sleep(1)
            logger.debug(f"当前状态 - Ctrl: {keyboard_manager.ctrl_pressed}, Alt: {keyboard_manager.alt_pressed}, 录音中: {is_recording}")
            
    except KeyboardInterrupt:
        pass
    finally:
        # 停止所有监听器
        debug_listener.stop()
        keyboard_manager.stop()
        print("\n调试结束")

if __name__ == "__main__":
    debug_keyboard_shortcuts()