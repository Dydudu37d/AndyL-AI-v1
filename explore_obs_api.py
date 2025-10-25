#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探索OBS WebSocket API的实际用法
用于确定obs-websocket-py库的正确使用方式
"""
import logging
from obswebsocket import obsws, requests

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# OBS WebSocket配置
HOST = '192.168.0.186'
PORT = 4455
PASSWORD = 'gR7UXLWyqEBaRd2S'

def explore_obs_api():
    """探索OBS WebSocket API的实际用法"""
    print("===== 探索OBS WebSocket API =====")
    
    # 显示requests模块的内容
    print("\n1. requests模块的内容:")
    print(f"类型: {type(requests)}")
    print(f"属性: {dir(requests)}")
    
    # 尝试创建连接
    print("\n2. 尝试创建OBS WebSocket连接...")
    try:
        ws = obsws(HOST, PORT, PASSWORD)
        ws.connect()
        print("✅ 连接成功!")
        
        # 显示obsws实例的方法
        print("\n3. obsws实例的方法:")
        methods = [method for method in dir(ws) if not method.startswith('__')]
        for method in methods:
            print(f"  - {method}")
            
        # 尝试直接调用一些可能的方法
        print("\n4. 尝试直接调用方法:")
        try:
            # 尝试获取场景列表的不同方法
            print("\n尝试获取场景列表...")
            
            # 方法1: 直接调用get_scene_list
            if hasattr(ws, 'get_scene_list'):
                print("尝试方法1: ws.get_scene_list()")
                result = ws.get_scene_list()
                print(f"结果: {result}")
            
            # 方法2: 尝试call方法的其他参数形式
            print("\n尝试方法2: 不同的call参数形式")
            # 尝试不同的参数组合
            try:
                result = ws.call("GetSceneList")
                print(f"  call(\"GetSceneList\")结果: {result}")
            except Exception as e:
                print(f"  call(\"GetSceneList\")失败: {e}")
            
            try:
                result = ws.call(request_type="GetSceneList")
                print(f"  call(request_type=\"GetSceneList\")结果: {result}")
            except Exception as e:
                print(f"  call(request_type=\"GetSceneList\")失败: {e}")
            
            try:
                # 尝试获取版本信息作为简单测试
                result = ws.call("GetVersion")
                print(f"  call(\"GetVersion\")结果: {result}")
            except Exception as e:
                print(f"  call(\"GetVersion\")失败: {e}")
                
        except Exception as e:
            print(f"调用方法时出错: {e}")
            import traceback
            traceback.print_exc()
            
        # 断开连接
        ws.disconnect()
        print("\n✅ 已断开连接")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    explore_obs_api()