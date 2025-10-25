import time
import signal
import sys
from microbit_communication import MicrobitCommunication

# 尝试导入集成模块
try:
    import wukong
    print("成功导入wukong模块")
except ImportError:
    print("警告: 无法导入wukong模块")
    wukong = None

try:
    import sonar
    print("成功导入sonar模块")
except ImportError:
    print("警告: 无法导入sonar模块")
    sonar = None

# 注意：不要直接导入microbit包，以避免循环导入问题

class WukongSonarMicrobit:
    def __init__(self, port=None, baudrate=115200):
        # 初始化配置，但不立即连接micro:bit
        self.port = port
        self.baudrate = baudrate
        self.microbit_comm = None
        
        # 初始化wukong和sonar模块
        self.initialize_wukong()
        self.initialize_sonar()
        
        # 设置信号处理，优雅退出
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        
    def initialize_wukong(self):
        """初始化wukong模块"""
        if wukong:
            try:
                # 这里是wukong模块的初始化代码
                # 根据wukong模块的实际API进行调整
                print("正在初始化wukong模块...")
                # 假设wukong模块提供了初始化函数
                # wukong.init()
                print("wukong模块初始化完成")
            except Exception as e:
                print(f"wukong模块初始化失败: {e}")
        else:
            print("wukong模块不可用")
    
    def initialize_sonar(self):
        """初始化sonar模块"""
        if sonar:
            try:
                # 这里是sonar模块的初始化代码
                # 根据sonar模块的实际API进行调整
                print("正在初始化sonar模块...")
                # 假设sonar模块提供了初始化函数
                # sonar.init()
                print("sonar模块初始化完成")
            except Exception as e:
                print(f"sonar模块初始化失败: {e}")
        else:
            print("sonar模块不可用")
    
    def run_demo(self):
        """运行集成演示"""
        print("开始运行wukong、sonar和micro:bit集成演示...")
        
        # 创建micro:bit通信实例
        self.microbit_comm = MicrobitCommunication(port=self.port, baudrate=self.baudrate)
        
        # 尝试连接到micro:bit，最多重试3次
        max_retries = 3
        retry_count = 0
        connected = False
        
        while retry_count < max_retries and not connected:
            if retry_count > 0:
                print(f"正在尝试第 {retry_count+1} 次连接...")
                time.sleep(1)  # 等待1秒后重试
            
            connected = self.microbit_comm.connect()
            retry_count += 1
        
        if not connected:
            print("无法连接到micro:bit，演示终止")
            print("请检查以下几点：")
            print("1. micro:bit是否正确连接到电脑USB端口")
            print("2. 端口是否被其他程序占用")
            print("3. 是否有足够的权限访问该端口")
            print("4. micro:bit是否刷入了正确的MicroPython程序")
            return
        
        try:
            # 显示欢迎信息
            self.microbit_comm.show_text("WUKONG+SONAR")
            time.sleep(2)
            
            # 主循环
            counter = 0
            while True:
                counter += 1
                
                # 发送计数器值到micro:bit
                self.microbit_comm.show_text(str(counter % 10))
                
                # 读取micro:bit数据
                data = self.microbit_comm.read_data()
                if data:
                    print(f"从micro:bit接收: {data}")
                    
                    # 示例：使用wukong处理接收到的数据
                    if wukong and "BUTTON" in data:
                        print("使用wukong处理按钮事件...")
                        # 这里是使用wukong模块处理数据的代码
                        # 例如: wukong.process_button_event(data)
                    
                    # 示例：使用sonar处理温度数据
                    if sonar and "TEMPERATURE" in data:
                        print("使用sonar处理温度数据...")
                        # 这里是使用sonar模块处理数据的代码
                        # 例如: sonar.analyze_temperature(data)
                
                # 每2秒获取一次温度
                if counter % 20 == 0:  # 每2秒(假设循环频率为10Hz)
                    temp_data = self.microbit_comm.get_temperature()
                    if temp_data:
                        print(f"当前温度: {temp_data}")
                
                # 每5秒显示一次图标
                if counter % 50 == 0:  # 每5秒
                    icons = ["HAPPY", "SURPRISED", "YES", "NO"]
                    icon_index = counter // 50 % len(icons)
                    self.microbit_comm.show_icon(icons[icon_index])
                
                time.sleep(0.1)  # 100ms采样间隔
                
        except Exception as e:
            print(f"演示过程中出错: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        self.microbit_comm.close()
        
        # 清理wukong和sonar资源
        if wukong:
            try:
                # wukong模块的清理代码
                pass
            except:
                pass
        
        if sonar:
            try:
                # sonar模块的清理代码
                pass
            except:
                pass
        
        print("资源清理完成")
    
    def handle_exit(self, sig, frame):
        """处理退出信号"""
        print("\n收到退出信号，正在停止演示...")
        # 直接退出，让run_demo中的finally块处理清理
        sys.exit(0)

if __name__ == "__main__":
    # 创建并运行集成演示
    try:
        integrated_demo = WukongSonarMicrobit(baudrate=115200)
        integrated_demo.run_demo()
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)
