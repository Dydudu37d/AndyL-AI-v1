#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过修改Python路径解决pyserial导入冲突的脚本
"""

import sys
import os

# 先保存原始的sys.path
original_path = sys.path.copy()

# 查找pyserial的安装路径
def find_pyserial_path():
    """查找pyserial的正确安装路径"""
    for path in sys.path:
        if path.endswith('.venv\Lib\site-packages') or path.endswith('.venv/Lib/site-packages'):
            pyserial_path = os.path.join(path, 'pyserial')
            if os.path.isdir(pyserial_path):
                print(f"找到pyserial目录: {pyserial_path}")
                return path
    
    # 如果找不到，返回空
    return None

# 尝试修改sys.path以优先使用pyserial
pyserial_site_packages = find_pyserial_path()
if pyserial_site_packages:
    # 从sys.path中移除所有可能包含冲突serial库的路径
    new_path = [pyserial_site_packages]
    for path in sys.path:
        if path != pyserial_site_packages:
            # 避免添加可能包含冲突serial库的路径
            if not ('serial' in path.lower() and not 'pyserial' in path.lower()):
                new_path.append(path)
    
    # 使用新的path
    sys.path = new_path
    print(f"修改后的Python路径:\n{sys.path}")

# 尝试导入pyserial库
try:
    import pyserial as serial
    print(f"成功导入pyserial库，版本: {serial.__version__}")
except ImportError:
    try:
        # 直接导入serial（这应该现在指向正确的pyserial）
        import serial
        print(f"成功导入serial库，模块路径: {serial.__file__}")
    except ImportError:
        print("错误: 无法导入pyserial或serial库")
        print("请重新安装pyserial库: pip install pyserial --upgrade --force-reinstall")
        sys.exit(1)

# 检查Serial和SerialException是否可用
if hasattr(serial, 'Serial') and hasattr(serial, 'SerialException'):
    print("Serial和SerialException类可用")
    
    # 导入必要的模块
    import time
    
    # 设置串口参数
    port = 'COM5'  # Arduino连接的端口
    baudrate = 9600
    timeout = 1
    
    ser = None
    try:
        # 尝试打开串口
        ser = serial.Serial(
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
            while True:
                # 检查是否有数据可读
                if ser.in_waiting > 0:
                    # 读取一行数据
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            print(f"从Arduino接收: {line}")
                    except Exception as e:
                        print(f"读取数据错误: {e}")
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.1)
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        print(f"请检查端口 {port} 是否可用，或者Arduino是否正确连接")
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
        # 打印详细的错误信息，有助于调试
        import traceback
        traceback.print_exc()
    finally:
        # 确保关闭串口
        if ser and ser.is_open:
            ser.close()
            print(f"已关闭串口 {port}")
else:
    print("错误: Serial或SerialException类不可用")
    print("当前serial模块的内容:")
    for attr in dir(serial):
        print(f"- {attr}")
    
    # 提供解决方法建议
    print("\n解决方法建议:")
    print("1. 卸载冲突的serial库: pip uninstall serial")
    print("2. 重新安装pyserial库: pip install pyserial --upgrade --force-reinstall")

# 恢复原始的sys.path
sys.path = original_path