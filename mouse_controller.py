import logging
from pynput import mouse, keyboard
import threading
import time

# 配置日志
logger = logging.getLogger("MouseController")
logger.setLevel(logging.INFO)

class MouseController:
    """鼠标控制器 - 提供鼠标和键盘控制功能"""
    
    def __init__(self):
        # 初始化鼠标控制器
        self.mouse_controller = mouse.Controller()
        # 初始化键盘控制器
        self.keyboard_controller = keyboard.Controller()
        # 控制标志
        self.is_enabled = False
        # 监听器
        self.listener = None
        self.listener_thread = None
        # 动作回调函数
        self.action_callback = None
        self.is_recording = False
        self.recorded_actions = []
        self.last_action_time = 0
        # 按键映射
        self.key_mapping = {
            'alt': keyboard.Key.alt,
            'alt_l': keyboard.Key.alt_l,
            'alt_r': keyboard.Key.alt_r,
            'alt_gr': keyboard.Key.alt_gr,
            'backspace': keyboard.Key.backspace,
            'caps_lock': keyboard.Key.caps_lock,
            'cmd': keyboard.Key.cmd,
            'cmd_l': keyboard.Key.cmd_l,
            'cmd_r': keyboard.Key.cmd_r,
            'ctrl': keyboard.Key.ctrl,
            'ctrl_l': keyboard.Key.ctrl_l,
            'ctrl_r': keyboard.Key.ctrl_r,
            'delete': keyboard.Key.delete,
            'down': keyboard.Key.down,
            'end': keyboard.Key.end,
            'enter': keyboard.Key.enter,
            'esc': keyboard.Key.esc,
            'f1': keyboard.Key.f1,
            'f2': keyboard.Key.f2,
            'f3': keyboard.Key.f3,
            'f4': keyboard.Key.f4,
            'f5': keyboard.Key.f5,
            'f6': keyboard.Key.f6,
            'f7': keyboard.Key.f7,
            'f8': keyboard.Key.f8,
            'f9': keyboard.Key.f9,
            'f10': keyboard.Key.f10,
            'f11': keyboard.Key.f11,
            'f12': keyboard.Key.f12,
            'home': keyboard.Key.home,
            'left': keyboard.Key.left,
            'page_down': keyboard.Key.page_down,
            'page_up': keyboard.Key.page_up,
            'right': keyboard.Key.right,
            'shift': keyboard.Key.shift,
            'shift_l': keyboard.Key.shift_l,
            'shift_r': keyboard.Key.shift_r,
            'space': keyboard.Key.space,
            'tab': keyboard.Key.tab,
            'up': keyboard.Key.up,
            'insert': keyboard.Key.insert,
            'menu': keyboard.Key.menu,
            'pause': keyboard.Key.pause,
            'print_screen': keyboard.Key.print_screen,
            'scroll_lock': keyboard.Key.scroll_lock,
            'num_lock': keyboard.Key.num_lock,
            'win': keyboard.Key.cmd
        }
    
    def initialize(self):
        """初始化控制器"""
        self.is_enabled = True
        logger.info("鼠标控制器初始化成功")
        return True
    
    def _convert_key(self, key):
        """
        将字符串按键名转换为pynput需要的按键对象
        
        参数:
            key: 按键名称（字符串或按键对象）
        
        返回:
            按键对象
        """
        if isinstance(key, keyboard.Key) or isinstance(key, keyboard.KeyCode):
            return key
        
        # 检查是否是特殊按键
        if key.lower() in self.key_mapping:
            return self.key_mapping[key.lower()]
        
        # 单个字符直接返回
        if len(key) == 1:
            return key.lower()
        
        # 如果是大写字母，可能需要结合shift
        if key.isupper():
            return key
        
        logger.warning(f"未知按键: {key}")
        return key
    
    def move_mouse(self, x, y, absolute=True, duration=0.0):
        """
        移动鼠标
        
        参数:
            x, y: 目标位置
            absolute: 是否使用绝对坐标
            duration: 移动持续时间（秒）- 不再使用，保持参数兼容性
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            if absolute:
                # 直接设置鼠标位置，不使用平滑移动
                self.mouse_controller.position = (int(x), int(y))
            else:
                self.mouse_controller.move(x, y)
            logger.debug(f"鼠标移动到: {x}, {y}")
            return True
        except Exception as e:
            logger.error(f"移动鼠标失败: {e}")
            return False
    
    def click(self, button=mouse.Button.left, count=1):
        """
        点击鼠标
        
        参数:
            button: 鼠标按钮 (left, right, middle)
            count: 点击次数
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            # 处理字符串按钮名
            if isinstance(button, str):
                if button.lower() == 'left':
                    button = mouse.Button.left
                elif button.lower() == 'right':
                    button = mouse.Button.right
                elif button.lower() == 'middle':
                    button = mouse.Button.middle
            
            for _ in range(count):
                self.mouse_controller.click(button)
                time.sleep(0.05)
            logger.debug(f"鼠标点击: {button}, 次数: {count}")
            return True
        except Exception as e:
            logger.error(f"点击鼠标失败: {e}")
            return False
    
    def scroll(self, x=0, y=0):
        """
        滚动鼠标滚轮
        
        参数:
            x: 水平滚动量
            y: 垂直滚动量
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            self.mouse_controller.scroll(x, y)
            logger.debug(f"鼠标滚动: x={x}, y={y}")
            return True
        except Exception as e:
            logger.error(f"滚动鼠标失败: {e}")
            return False
    
    def press_key(self, key):
        """
        按下键盘按键
        
        参数:
            key: 按键
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            converted_key = self._convert_key(key)
            self.keyboard_controller.press(converted_key)
            logger.debug(f"按下按键: {key}")
            return True
        except Exception as e:
            logger.error(f"按下按键失败: {e}")
            return False
    
    def release_key(self, key):
        """
        释放键盘按键
        
        参数:
            key: 按键
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            converted_key = self._convert_key(key)
            self.keyboard_controller.release(converted_key)
            logger.debug(f"释放按键: {key}")
            return True
        except Exception as e:
            logger.error(f"释放按键失败: {e}")
            return False
    
    def type_string(self, text, delay=0.05):
        """
        输入字符串
        
        参数:
            text: 要输入的文本
            delay: 每个字符之间的延迟（秒）
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            self.keyboard_controller.type(text)
            logger.debug(f"输入文本: {text}")
            return True
        except Exception as e:
            logger.error(f"输入文本失败: {e}")
            # 尝试逐个字符输入
            try:
                for char in text:
                    self.keyboard_controller.press(char)
                    self.keyboard_controller.release(char)
                    time.sleep(delay)
                logger.debug(f"通过逐个字符输入文本: {text}")
                return True
            except Exception as e2:
                logger.error(f"逐个字符输入文本也失败: {e2}")
                return False
    
    def press_combination(self, keys):
        """
        按下组合键
        
        参数:
            keys: 按键列表，如['ctrl', 'c']
        """
        if not self.is_enabled:
            logger.warning("控制器未启用")
            return False
        
        try:
            # 转换所有按键
            converted_keys = [self._convert_key(key) for key in keys]
            
            # 按下所有键
            for key in converted_keys:
                self.keyboard_controller.press(key)
            
            # 短暂延迟
            time.sleep(0.1)
            
            # 释放所有键（反向顺序）
            for key in reversed(converted_keys):
                self.keyboard_controller.release(key)
            
            logger.debug(f"按下组合键: {keys}")
            return True
        except Exception as e:
            logger.error(f"按下组合键失败: {e}")
            # 尝试释放所有已按下的键
            try:
                for key in keys:
                    try:
                        converted_key = self._convert_key(key)
                        self.keyboard_controller.release(converted_key)
                    except:
                        pass
            except:
                pass
            return False
    
    def get_mouse_position(self):
        """
        获取当前鼠标位置
        
        返回:
            (x, y) 坐标元组
        """
        try:
            pos = self.mouse_controller.position
            logger.debug(f"获取鼠标位置: {pos}")
            return pos
        except Exception as e:
            logger.error(f"获取鼠标位置失败: {e}")
            return (0, 0)
    
    def start_monitoring(self, callback=None):
        """
        开始监控用户的鼠标和键盘操作
        
        参数:
            callback: 当检测到操作时的回调函数
        """
        if self.listener:
            logger.warning("监控已经在运行")
            return False
        
        self.action_callback = callback
        
        def on_move(x, y):
            self._handle_action("move", x=x, y=y)
        
        def on_click(x, y, button, pressed):
            self._handle_action("click", x=x, y=y, button=button, pressed=pressed)
        
        def on_scroll(x, y, dx, dy):
            self._handle_action("scroll", x=x, y=y, dx=dx, dy=dy)
        
        def on_press(key):
            self._handle_action("key_press", key=key)
        
        def on_release(key):
            self._handle_action("key_release", key=key)
        
        try:
            # 创建鼠标监听器
            mouse_listener = mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll
            )
            
            # 创建键盘监听器
            keyboard_listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
            
            # 在单独的线程中启动监听器
            self.listener_thread = threading.Thread(target=self._run_listeners, 
                                                   args=(mouse_listener, keyboard_listener))
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            logger.info("开始监控鼠标和键盘操作")
            return True
        except Exception as e:
            logger.error(f"启动监控失败: {e}")
            return False
    
    def _run_listeners(self, mouse_listener, keyboard_listener):
        """在单独线程中运行监听器"""
        mouse_listener.start()
        keyboard_listener.start()
        
        # 保持线程运行
        try:
            while self.is_enabled:
                time.sleep(0.1)
        finally:
            mouse_listener.stop()
            keyboard_listener.stop()
    
    def _handle_action(self, action_type, **kwargs):
        """处理用户操作"""
        current_time = time.time()
        
        # 防止事件过于频繁
        if current_time - self.last_action_time < 0.01:
            return
        
        self.last_action_time = current_time
        
        # 如果有回调函数，调用它
        if self.action_callback:
            try:
                self.action_callback(action_type, **kwargs)
            except Exception as e:
                logger.error(f"执行回调函数失败: {e}")
        
        # 如果正在录制，保存动作
        if self.is_recording:
            self.recorded_actions.append({
                "type": action_type,
                "time": time.time(),
                "params": kwargs
            })
    
    def stop_monitoring(self):
        """停止监控用户的鼠标和键盘操作"""
        if not self.listener_thread:
            return
        
        self.is_enabled = False
        
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        
        self.listener_thread = None
        logger.info("停止监控鼠标和键盘操作")
    
    def start_recording(self):
        """开始录制用户操作"""
        self.is_recording = True
        self.recorded_actions = []
        logger.info("开始录制用户操作")
    
    def stop_recording(self):
        """停止录制用户操作"""
        self.is_recording = False
        logger.info(f"停止录制用户操作，共录制 {len(self.recorded_actions)} 个动作")
        return self.recorded_actions
    
    def play_recording(self, recorded_actions=None):
        """播放录制的用户操作"""
        actions = recorded_actions if recorded_actions else self.recorded_actions
        
        if not actions:
            logger.warning("没有可播放的录制操作")
            return False
        
        logger.info(f"开始播放录制的操作，共 {len(actions)} 个动作")
        
        # 获取第一个动作的时间作为基准
        base_time = actions[0]["time"]
        
        try:
            for i, action in enumerate(actions):
                # 计算动作之间的延迟
                if i > 0:
                    delay = action["time"] - actions[i-1]["time"]
                    time.sleep(delay)
                
                # 执行动作
                self._execute_recorded_action(action)
            
            logger.info("播放录制的操作完成")
            return True
        except Exception as e:
            logger.error(f"播放录制的操作失败: {e}")
            return False
    
    def _execute_recorded_action(self, action):
        """执行录制的单个动作"""
        action_type = action["type"]
        params = action["params"]
        
        try:
            if action_type == "move":
                self.move_mouse(params["x"], params["y"])
            elif action_type == "click" and params["pressed"]:
                self.click(params["button"])
            elif action_type == "scroll":
                self.scroll(params["dx"], params["dy"])
            elif action_type == "key_press":
                self.press_key(params["key"])
            elif action_type == "key_release":
                self.release_key(params["key"])
        except Exception as e:
            logger.error(f"执行录制动作失败: {e}")
    
    def shutdown(self):
        """关闭控制器"""
        self.stop_monitoring()
        self.is_enabled = False
        logger.info("鼠标控制器已关闭")

# 创建全局控制器实例
_controller_instance = None

# 用于获取全局控制器实例的工厂函数
def get_mouse_controller():
    """获取鼠标控制器实例"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = MouseController()
    return _controller_instance

# 测试函数
def test_mouse_controller():
    """测试鼠标控制器功能"""
    # 配置日志
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("======= 鼠标控制器测试 =======")
    print("此测试将演示鼠标和键盘控制功能")
    print("按Enter键继续...")
    input()
    
    # 获取控制器实例
    controller = get_mouse_controller()
    controller.initialize()
    
    try:
        # 测试鼠标移动
        print("\n测试1: 移动鼠标到屏幕中央")
        screen_width, screen_height = 1920, 1080  # 假设屏幕分辨率
        controller.move_mouse(screen_width // 2, screen_height // 2)
        time.sleep(1)
        
        # 测试鼠标点击
        print("\n测试2: 鼠标左键点击")
        controller.click()
        time.sleep(1)
        
        # 测试键盘输入
        print("\n测试3: 键盘输入文本")
        controller.type_string("Hello, 这是鼠标控制器测试!")
        time.sleep(1)
        
        # 测试组合键
        print("\n测试4: 按下Ctrl+C组合键")
        controller.press_combination(["ctrl", "c"])
        
        print("\n测试完成！")
    finally:
        controller.shutdown()

if __name__ == "__main__":
    test_mouse_controller()