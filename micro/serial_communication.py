import time
import signal
import wukong
import sys

# 解决serial库冲突问题
try:
    # 直接尝试从pyserial库导入所需的类
    from pyserial import Serial, SerialException
    print("成功从pyserial库导入Serial和SerialException类")
except ImportError:
    # 如果直接导入pyserial失败，尝试找到正确的导入路径
    try:
        # 检查serial.tools.list_ports是否可用
        from serial.tools import list_ports
        print("找到serial.tools.list_ports模块")
        
        # 尝试通过不同路径导入Serial和SerialException
        try:
            from serial.serialutil import SerialException
            from serial.serialwin32 import Serial
            print("成功从serial.serialwin32导入Serial和从serial.serialutil导入SerialException")
        except ImportError:
            print("错误: 无法找到Serial或SerialException类")
            print("请检查您的Python环境中安装的serial/pyserial库版本")
            sys.exit(1)
    except ImportError:
        print("错误: 请确保已安装pyserial库")
        print("您可以使用命令 'pip install pyserial' 来安装")
        print("注意: 您的环境中可能存在冲突的serial库，请考虑卸载它")
        sys.exit(1)

class SerialCommunication:
    def __init__(self, port='COM3', baudrate=9600, timeout=1):
        """初始化串口通信类"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.running = False
        
        # 设置信号处理，优雅退出
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
    
    def connect(self):
        """建立串口连接"""
        try:
            self.ser = Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity='N',  # PARITY_NONE
                stopbits=1,  # STOPBITS_ONE
                bytesize=8   # EIGHTBITS
            )
            if self.ser.is_open:
                print(f"成功连接到 {self.port}")
                self.running = True
                return True
            return False
        except SerialException as e:
            print(f"串口连接失败: {e}")
            print(f"请检查端口 {self.port} 是否可用，或者Arduino是否正确连接")
            return False
    
    def send_data(self, data):
        """发送数据到Arduino"""
        if not self.ser or not self.ser.is_open:
            print("错误: 串口未连接")
            return False
        
        try:
            if isinstance(data, str):
                self.ser.write(data.encode('utf-8'))
            else:
                self.ser.write(data)
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False
    
    def read_data(self):
        """读取Arduino发送的数据"""
        if not self.ser or not self.ser.is_open:
            print("错误: 串口未连接")
            return None
        
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                return line
            return None
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None
    
    def run(self):
        """运行主循环"""
        if not self.connect():
            return
        
        try:
            # 发送初始命令
            self.send_data(b'H')  # 发送'H'来打开LED
            print("已发送初始命令'H'到Arduino")
            
            print("开始接收数据，按Ctrl+C退出...")
            while self.running:
                data = self.read_data()
                if data:
                    print(f"从Arduino接收: {data}")
                
                # 定期发送心跳包，保持连接活跃
                # self.send_data(b'.')
                
                time.sleep(0.1)  # 100ms采样间隔
                
        except Exception as e:
            print(f"运行时错误: {e}")
        finally:
            self.close()
    
    def close(self):
        """关闭串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"已关闭串口 {self.port}")
    
    def handle_exit(self, sig, frame):
        """处理退出信号"""
        print("\n收到退出信号，正在关闭程序...")
        self.running = False
        time.sleep(0.5)  # 给系统一点时间清理资源
        sys.exit(0)

if __name__ == "__main__":
    # 检查系统平台
    is_windows = sys.platform.startswith('win')
    default_port = 'COM5' if is_windows else '/dev/ttyACM0'
    
    # 创建并运行串口通信实例
    serial_comm = SerialCommunication(port=default_port, baudrate=9600)
    serial_comm.run()