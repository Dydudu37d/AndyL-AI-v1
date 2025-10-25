#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版OBS控制器
基于OBS WebSocket 5.x API，提供最基本的OBS控制功能
"""
import logging
import time
from obswebsocket import obsws

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimpleOBSController')


class SimpleOBSController:
    """简化版OBS控制器 - 提供基本的OBS控制功能"""
    
    def __init__(self, host='localhost', port=4455, password='', auto_connect=True):
        """
        初始化OBS控制器
        
        Args:
            host (str): OBS WebSocket服务器地址
            port (int): OBS WebSocket服务器端口
            password (str): OBS WebSocket服务器密码
            auto_connect (bool): 是否自动连接
        """
        self.host = host
        self.port = port
        self.password = password
        self.ws = None
        self.is_connected = False
        
        if auto_connect:
            self.connect()
    
    def connect(self):
        """连接到OBS WebSocket服务器"""
        try:
            logger.info(f"尝试连接到OBS WebSocket服务器: {self.host}:{self.port}")
            self.ws = obsws(self.host, self.port, self.password)
            self.ws.connect()
            self.is_connected = True
            logger.info("OBS WebSocket连接成功!")
            return True
        except Exception as e:
            logger.error(f"OBS WebSocket连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开与OBS WebSocket服务器的连接"""
        try:
            if self.ws and self.is_connected:
                self.ws.disconnect()
                self.is_connected = False
                logger.info("OBS WebSocket连接已断开")
            return True
        except Exception as e:
            logger.error(f"断开OBS WebSocket连接时出错: {e}")
            return False
    
    def reconnect(self, max_attempts=3, delay=2):
        """重新连接到OBS WebSocket服务器"""
        self.disconnect()
        
        for attempt in range(max_attempts):
            logger.info(f"重新连接尝试 {attempt+1}/{max_attempts}")
            if self.connect():
                return True
            logger.info(f"等待{delay}秒后重试...")
            time.sleep(delay)
        
        logger.error(f"达到最大重连次数({max_attempts})，连接失败")
        return False
    
    def _check_connection(self):
        """检查连接状态，如果断开则尝试重连"""
        if not self.is_connected:
            logger.warning("OBS WebSocket连接已断开，尝试重连...")
            return self.reconnect()
        return True
    
    def call_request(self, request_type, **kwargs):
        """
        通用请求调用方法
        尝试使用不同的方式发送请求到OBS
        
        Args:
            request_type (str): 请求类型字符串
            **kwargs: 额外的请求参数
        
        Returns:
            响应结果或None
        """
        try:
            if not self._check_connection():
                return None
            
            logger.debug(f"尝试发送请求: {request_type}, 参数: {kwargs}")
            
            # 注意：这里我们不直接调用ws.call方法
            # 因为我们还没有确定正确的API用法
            # 我们会在测试脚本中进一步探索
            
            # 目前只返回连接状态
            return True
        except Exception as e:
            logger.error(f"发送请求时出错: {e}")
            return None


# 创建全局控制器实例
simple_obs_controller = SimpleOBSController('192.168.0.186', 4455, 'gR7UXLWyqEBaRd2S')


if __name__ == "__main__":
    """测试简化版OBS控制器"""
    print("===== 测试简化版OBS控制器 =====")
    
    # 测试连接
    if simple_obs_controller.is_connected:
        print("✅ OBS WebSocket连接成功!")
    else:
        print("❌ OBS WebSocket连接失败!")
        
    # 打印控制器信息
    print(f"\n控制器信息:")
    print(f"  主机: {simple_obs_controller.host}")
    print(f"  端口: {simple_obs_controller.port}")
    print(f"  连接状态: {'已连接' if simple_obs_controller.is_connected else '未连接'}")
    
    # 提示用户这是一个基础版本
    print("\n注意：这是一个基础版本的OBS控制器，")
    print("完整的功能需要进一步探索OBS WebSocket API的正确用法。")
    print("建议查看OBS WebSocket插件的官方文档以了解更多信息。")