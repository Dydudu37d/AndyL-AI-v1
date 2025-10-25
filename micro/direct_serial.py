#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从正确的模块路径导入Serial和SerialException的脚本
绕过了导入冲突问题
"""

import sys
import importlib

# 尝试通过动态导入找到正确的模块路径
def find_serial_class():
    """动态查找Serial类的正确导入路径"""
    # 可能的导入路径列表
    possible_paths = [
        'pyserial.serial',
        'serial.serial',
        'serial.serialwin32',
        'pyserial.serialwin32',
        'serial.serialposix',
        'pyserial.serialposix'
    ]
    
    # 可能的异常类导入路径
    exception_paths = [
        'serial.serialutil',
        'pyserial.serialutil'
    ]
    
    Serial = None
    SerialException = None
    
    # 查找SerialException
    for path in exception_paths:
        try:
            module = importlib.import_module(path)
            if hasattr(module, 'SerialException'):
                SerialException = module.SerialException
                print(f"找到SerialException在: {path}")
                break
        except ImportError:
            continue
    
    # 查找Serial类
    for path in possible_paths:
        try:
            module = importlib.import_module(path)
            if hasattr(module, 'Serial'):
                Serial = module.Serial
                print(f"找到Serial类在: {path}")
                break
        except ImportError:
            continue
    
    return Serial, SerialException

def main():
    """主函数"""
    # 查找Serial和SerialException类
    Serial, SerialException = find_serial_class()
    
    if not Serial or not SerialException:
        print("错误: 无法找到Serial或SerialException类")
        print("请检查您的Python环境中的pyserial库")
        print("\n解决方法建议:")
        print("1. 卸载冲突的serial库: pip uninstall serial")
        print("2. 重新安装pyserial库: pip install pyserial --upgrade")
        sys.exit(1)
    
    # 设置串口参数
    port = 'COM5'  # 根据之前的测试，这是Arduino连接的端口
    baudrate = 9600
    timeout = 1
    
    ser = None
    try:
        # 尝试打开串口
        ser = Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        
        if ser.is_open:
            print(f"成功连接到 {port} (波特率: {baudrate})")
            
            # 发送初始命令
            ser.write(b'H')
            print("已发送初始命令'H'到Arduino")
            
            print("开始接收数据，按Ctrl+C退出...")
            
            # 主循环
            import time
            while True:
                # 检查是否有数据可读
                if hasattr(ser, 'in_waiting') and ser.in_waiting > 0:
                    # 读取一行数据
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            print(f"从Arduino接收: {line}")
                    except Exception as e:
                        print(f"读取数据错误: {e}")
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.1)
    except SerialException as e:
        print(f"串口连接失败: {e}")
        print(f"请检查端口 {port} 是否可用，或者Arduino是否正确连接")
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保关闭串口
        if ser and hasattr(ser, 'close') and hasattr(ser, 'is_open') and ser.is_open:
            ser.close()
            print(f"已关闭串口 {port}")

if __name__ == "__main__":
    main()