#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快捷键功能验证脚本

此脚本用于测试修复后的键盘快捷键功能，特别是Ctrl+Alt+R组合键。
它将显示键盘监听器的状态，并在检测到快捷键时提供视觉反馈。
"""

import os
import sys
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShortcutVerifier")

# 尝试导入键盘快捷键模块
try:
    from keyboard_shortcuts import (
        start_keyboard_shortcuts,
        stop_keyboard_shortcuts,
        set_recording_callback,
        get_keyboard_listener_status
    )
    print("✅ 成功导入键盘快捷键模块")
except ImportError as e:
    print(f"❌ 导入键盘快捷键模块失败: {e}")
    print("💡 请确保脚本位于正确的目录中")
    sys.exit(1)

# 回调函数，当检测到Ctrl+Alt+R时调用
def on_shortcut_triggered():
    """检测到快捷键时的回调函数"""
    print("\n🎉🎉🎉 成功检测到 Ctrl+Alt+R 快捷键!")
    print("✅ 修饰键指令功能正常工作!")
    logger.info("Ctrl+Alt+R快捷键已触发")
    
    # 显示一些视觉反馈
    for i in range(3):
        sys.stdout.write("✨ ")
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write("\n")

# 显示系统信息
def display_system_info():
    """显示系统和键盘监听器信息"""
    try:
        import platform
        print("\n=== 系统信息 ===")
        print(f"操作系统: {platform.system()} {platform.release()}")
        print(f"Python版本: {platform.python_version()}")
        
        # 获取键盘监听器状态
        status = get_keyboard_listener_status()
        print("\n=== 键盘监听器状态 ===")
        print(f"可用: {status['available']}")
        print(f"运行中: {status['running']}")
        print(f"平台: {status['platform']}")
        print(f"需要管理员权限: {status['needs_admin']}")
        
        # 提示用户关于管理员权限的重要性
        if status['platform'] == "Windows" and status['needs_admin']:
            print("\n⚠️  重要提示:")
            print("  在Windows系统上，键盘快捷键功能通常需要管理员权限才能正常工作。")
            print("  如果快捷键无法被检测到，请尝试以管理员身份重新运行此脚本。")
            
            # 尝试一种简单的管理员权限检查
            try:
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
                print(f"  当前权限状态: {'管理员' if is_admin else '标准用户'}")
            except Exception:
                print("  无法确定当前权限状态")
    except Exception as e:
        logger.error(f"显示系统信息时出错: {e}")

# 主函数
def main():
    """主函数"""
    print("====================================")
    print("       键盘快捷键功能验证工具       ")
    print("====================================")
    print("此工具用于测试修复后的键盘快捷键功能")
    print("\n使用说明:")
    print("  1. 请按下 Ctrl+Alt+R 组合键")
    print("  2. 如果检测到快捷键，将显示成功消息")
    print("  3. 按 Ctrl+C 退出程序")
    
    # 显示系统信息
    display_system_info()
    
    try:
        # 设置回调函数
        set_recording_callback(on_shortcut_triggered)
        print("\n✅ 回调函数已设置")
        
        # 启动键盘快捷键监听
        print("\n正在启动键盘快捷键监听...")
        start_keyboard_shortcuts()
        print("✅ 键盘快捷键监听已启动")
        
        # 显示当前状态
        status = get_keyboard_listener_status()
        print(f"\n当前状态: {status}")
        
        print("\n====================================")
        print("等待快捷键触发... (按Ctrl+C退出)")
        print("====================================")
        
        # 保持程序运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n接收到中断信号，正在退出...")
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"\n❌ 程序运行出错: {e}")
    finally:
        # 停止键盘监听
        try:
            stop_keyboard_shortcuts()
            print("✅ 键盘快捷键监听已停止")
        except Exception:
            pass
        
    print("\n====================================")
    print("测试完成")
    print("====================================")

if __name__ == "__main__":
    main()