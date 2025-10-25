#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探索OBS WebSocket的base_classes
专门针对OBS WebSocket 5.x版本API的使用方式
"""
import logging
import json
from obswebsocket import obsws
from obswebsocket.core import base_classes

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OBSBaseClassExplorer')

# OBS WebSocket配置
HOST = '192.168.0.186'
PORT = 4455
PASSWORD = 'gR7UXLWyqEBaRd2S'


def explore_base_classes():
    """探索OBS WebSocket的base_classes"""
    print("===== 探索OBS WebSocket的base_classes =====")
    
    # 1. 详细检查base_classes模块
    print("\n1. 详细检查base_classes模块:")
    print(f"类型: {type(base_classes)}")
    print(f"模块内容: {dir(base_classes)}")
    
    # 查看每个类和函数
    for attr_name in dir(base_classes):
        if not attr_name.startswith('__'):
            attr = getattr(base_classes, attr_name)
            print(f"\n{attr_name}:")
            print(f"  类型: {type(attr)}")
            
            if hasattr(attr, '__doc__') and attr.__doc__:
                print(f"  文档: {attr.__doc__.strip().split('\n')[0]}")
            
            if hasattr(attr, '__mro__'):
                print(f"  父类: {[cls.__name__ for cls in attr.__mro__]}")
            
            if callable(attr):
                print(f"  可调用")
            else:
                print(f"  值: {attr}")
    
    # 2. 连接到OBS并尝试基于5.x API格式发送请求
    print("\n\n===== 尝试发送OBS WebSocket 5.x API请求 =====")
    
    ws = None
    try:
        logger.info(f"连接到OBS WebSocket服务器: {HOST}:{PORT}")
        ws = obsws(HOST, PORT, PASSWORD)
        ws.connect()
        logger.info("✅ 连接成功!")
        
        # 检查是否有call方法
        if hasattr(ws, 'call'):
            print("\n2. 尝试使用ws.call方法:")
            
            # OBS WebSocket 5.x API使用JSON格式的请求
            # 尝试构造一个简单的GetSceneList请求
            try:
                # 方式1: 尝试直接传递字符串
                print("  方式1: 尝试直接传递请求类型字符串 'GetSceneList'")
                result = ws.call("GetSceneList")
                print(f"  成功: {result}")
            except Exception as e:
                print(f"  方式1失败: {e}")
                
            try:
                # 方式2: 尝试传递字典对象
                print("  方式2: 尝试传递请求字典 {\"requestType\": \"GetSceneList\"}")
                request_data = {"requestType": "GetSceneList"}
                result = ws.call(request_data)
                print(f"  成功: {result}")
            except Exception as e:
                print(f"  方式2失败: {e}")
                
            try:
                # 方式3: 尝试传递JSON字符串
                print("  方式3: 尝试传递JSON字符串 '{\"requestType\": \"GetSceneList\"}'")
                request_json = json.dumps({"requestType": "GetSceneList"})
                result = ws.call(request_json)
                print(f"  成功: {result}")
            except Exception as e:
                print(f"  方式3失败: {e}")
        else:
            print("\nws对象没有call方法")
        
        # 3. 查看是否能直接访问websocket对象
        if hasattr(ws, 'ws') and ws.ws:
            print("\n3. 尝试直接使用底层websocket对象:")
            print(f"  websocket类型: {type(ws.ws)}")
            
            # 尝试直接发送OBS WebSocket 5.x格式的请求
            try:
                request_data = {
                    "op": 6,  # OBS WebSocket 5.x中的Request操作码
                    "d": {
                        "requestType": "GetSceneList",
                        "requestId": "test-request-1"
                    }
                }
                request_json = json.dumps(request_data)
                print(f"  尝试发送请求: {request_json}")
                ws.ws.send(request_json)
                print("  请求已发送")
                
                # 尝试接收响应
                # 注意：这可能会阻塞，取决于OBS的响应时间
                try:
                    response = ws.ws.recv()
                    print(f"  收到响应: {response}")
                except Exception as e:
                    print(f"  接收响应失败: {e}")
            except Exception as e:
                print(f"  发送请求失败: {e}")
        
        # 总结
        print("\n\n===== 探索总结 =====")
        print("1. 成功连接到OBS WebSocket 5.5.4服务器")
        print("2. 我们了解了base_classes模块的结构")
        print("3. 尝试了多种方式发送请求，但需要进一步了解正确的API用法")
        print("\n建议查看OBS WebSocket 5.x版本的官方API文档，")
        print("特别是关于请求格式和响应处理的部分。")
        
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}")
    finally:
        # 断开连接
        if ws:
            try:
                ws.disconnect()
                logger.info("✅ 已断开连接")
            except:
                pass


if __name__ == "__main__":
    explore_base_classes()