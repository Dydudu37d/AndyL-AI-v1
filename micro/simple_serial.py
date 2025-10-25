#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Arduino串口通信程序
解决了原始脚本中的导入冲突问题
"""

import time
import sys
import os

# 确保使用正确的pyserial库
try:
    # 直接导入pyserial模块
    import pyserial as serial
    print(f"成功导入pyserial库，版本: {serial.VERSION}")
except ImportError:
    try:
        # 如果直接导入pyserial失败，尝试导入serial
        import serial
        print(f"成功导入serial库，模块路径: {serial.__file__}")
    except ImportError:
        print("错误: 请安装pyserial库")
        print("您可以使用命令 'pip install pyserial' 来安装")
        sys.exit(1)

def list_available_ports():
    """列出所有可用的串口端口"""
    try:
        from serial.tools import list_ports
        ports = list_ports.comports()
        if not ports:
            print("没有找到可用的串口端口")
            return []
        
        print("可用的串口端口:")
        for i, port in enumerate(ports):
            print(f"{i+1}. {port.device}: {port.description}")
        
        return [port.device for port in ports]
    except ImportError:
        print("无法导入serial.tools.list_ports模块")
        return []

def main():
    """主函数"""
    # 检查系统平台
    is_windows = sys.platform.startswith('win')
    default_port = 'COM5' if is_windows else '/dev/ttyACM0'
    
    # 列出可用的端口
    available_ports = list_available_ports()
    
    # 如果默认端口不在可用端口列表中，让用户选择
    port = default_port
    if available_ports and port not in available_ports:
        print(f"默认端口 {port} 不可用")
        try:
            choice = input(f"请选择端口 (1-{len(available_ports)}): ")
            port = available_ports[int(choice) - 1]
        except (ValueError, IndexError):
            print("无效的选择，使用第一个可用端口")
            port = available_ports[0]
    
    # 设置串口参数
    baudrate = 9600
    timeout = 1
    
    ser = None
    try:
        # 打开串口
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        
        if ser.is_open:
            print(f"成功连接到 {port} (波特率: {baudrate})")
            
            # 发送初始命令
            ser.write(b'H')
            print("已发送初始命令'H'到Arduino")
            
            print("开始接收数据，按Ctrl+C退出...")
            while True:
                # 检查是否有数据可读
                if ser.in_waiting > 0:
                    # 读取一行数据
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"从Arduino接收: {line}")
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.1)
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        print(f"请检查端口 {port} 是否可用，或者Arduino是否正确连接")
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        # 确保关闭串口
        if ser and ser.is_open:
            ser.close()
            print(f"已关闭串口 {port}")

if __name__ == "__main__":
    main()