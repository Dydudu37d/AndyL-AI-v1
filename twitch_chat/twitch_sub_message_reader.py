#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Twitch订阅文字消息读取示例
此脚本演示如何通过IRC和EventSub两种方式读取Twitch订阅的文字消息
"""

import os
import sys
import json
import logging
import requests
import socket
import http.server
import socketserver
import urllib.parse
from dotenv import load_dotenv

# 配置日志
def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('twitch_sub_message.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class TwitchSubMessageReader:
    """Twitch订阅文字消息读取器"""
    def __init__(self):
        """初始化"""
        self.logger = setup_logging()
        
        # 加载环境变量
        load_dotenv()
        
        # 获取环境变量
        self.client_id = os.getenv('TWITCH_ID')
        self.client_secret = os.getenv('TWITCH_SECRET')
        self.oauth_token = os.getenv('TWITCH_OAUTH_TOKEN')
        
        # 验证必要的环境变量
        if not self.client_id or not self.oauth_token:
            self.logger.error("请设置TWITCH_ID和TWITCH_OAUTH_TOKEN环境变量")
            sys.exit(1)
        
        # 用户信息
        self.user_id = None
        self.user_name = None
        
        # 获取用户信息
        self._get_user_info()
    
    def _get_user_info(self):
        """使用OAuth令牌获取用户信息"""
        try:
            url = "https://api.twitch.tv/helix/users"
            headers = {
                'Client-ID': self.client_id,
                'Authorization': f'Bearer {self.oauth_token}'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data['data'] and len(data['data']) > 0:
                    self.user_id = data['data'][0]['id']
                    self.user_name = data['data'][0]['login']
                    self.logger.info(f"已获取用户信息: ID={self.user_id}, 用户名={self.user_name}")
                else:
                    self.logger.error("未能获取用户信息，响应数据为空")
            else:
                self.logger.error(f"获取用户信息失败，状态码: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
        except Exception as e:
            self.logger.error(f"获取用户信息时发生异常: {e}")
    
    def read_sub_messages_via_irc(self):
        """通过IRC方式读取订阅文字消息"""
        self.logger.info("正在通过IRC方式连接到Twitch...")
        
        # 确保已获取用户信息
        if not self.user_id or not self.user_name:
            self.logger.error("无法连接IRC，因为未能获取用户信息")
            return False
        
        try:
            # 连接到Twitch IRC服务器
            irc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            irc.connect(('irc.chat.twitch.tv', 6667))
            
            # 发送认证信息
            nick = self.user_name
            irc.send(f"PASS oauth:{self.oauth_token}\r\n".encode('utf-8'))
            irc.send(f"NICK {nick}\r\n".encode('utf-8'))
            irc.send(f"JOIN #{self.user_name}\r\n".encode('utf-8'))
            
            self.logger.info(f"已连接到 {self.user_name} 的聊天室")
            self.logger.info("正在监听订阅通知... (按Ctrl+C退出)")
            print("\n====================================")
            print("开始监听订阅消息 (按Ctrl+C退出)")
            print("====================================")
            
            # 监听聊天消息
            while True:
                # 接收数据
                data = irc.recv(2048).decode('utf-8')
                
                # 回复PING请求以保持连接
                if data.startswith('PING'):
                    irc.send("PONG\r\n".encode('utf-8'))
                
                # 处理订阅通知
                if 'USERNOTICE' in data and 'subscriber' in data:
                    self._parse_and_display_irc_sub_message(data)
                
        except KeyboardInterrupt:
            self.logger.info("正在退出IRC连接...")
            return True
        except Exception as e:
            self.logger.error(f"IRC连接发生异常: {e}")
            return False
    
    def _parse_and_display_irc_sub_message(self, data):
        """解析并显示IRC订阅消息"""
        try:
            # 解析订阅信息
            parts = data.split(';')
            
            # 获取订阅者名称
            subscriber_name = None
            for part in parts:
                if part.startswith('display-name='):
                    subscriber_name = part.split('=')[1]
                    break
            
            # 获取订阅等级
            sub_tier = '1000'  # 默认等级1
            for part in parts:
                if part.startswith('msg-param-sub-plan='):
                    sub_tier = part.split('=')[1]
                    break
            
            # 将等级转换为可读格式
            tier_map = {'1000': 'Tier 1', '2000': 'Tier 2', '3000': 'Tier 3', 'Prime': 'Prime'}
            sub_tier_readable = tier_map.get(sub_tier, f'Unknown ({sub_tier})')
            
            # 获取订阅消息
            message = None
            if ':' in data:
                message_parts = data.split(':', 2)
                if len(message_parts) > 2:
                    message = message_parts[2].strip()
            
            # 显示订阅信息
            print("\n🎉 收到新订阅!")
            print(f"  订阅者: {subscriber_name}")
            print(f"  等级: {sub_tier_readable}")
            if message:
                print(f"  订阅消息: {message}")
            print("====================================")
            
            # 记录到日志
            self.logger.info(f"收到订阅: {subscriber_name} (等级: {sub_tier_readable})")
            if message:
                self.logger.info(f"订阅消息: {message}")
            
        except Exception as e:
            self.logger.error(f"解析订阅消息时发生异常: {e}")
    
    def read_sub_messages_via_eventsub(self, callback_url, server_port=8080):
        """通过EventSub方式读取订阅文字消息"""
        self.logger.info("正在通过EventSub方式设置Webhook...")
        
        # 确保用户信息已获取
        if not self.user_id or not self.user_name:
            self.logger.error("无法设置EventSub Webhook，因为未能获取用户信息")
            return False
        
        # 确保已设置client_secret
        if not self.client_secret:
            self.logger.error("使用EventSub方式需要设置TWITCH_SECRET环境变量")
            return False
        
        try:
            # 1. 获取应用访问令牌
            app_access_token = self._get_app_access_token()
            if not app_access_token:
                self.logger.error("无法获取应用访问令牌，无法设置EventSub Webhook")
                return False
            
            # 2. 创建订阅事件
            subscription_success = self._create_subscription_event(callback_url, app_access_token)
            if not subscription_success:
                self.logger.error("创建订阅事件失败")
                return False
            
            # 3. 启动本地服务器接收回调
            print("\n====================================")
            print(f"本地服务器已启动，监听端口: {server_port}")
            print("等待接收订阅消息... (按Ctrl+C退出)")
            print("====================================")
            self._start_eventsub_server(server_port)
            
        except Exception as e:
            self.logger.error(f"设置EventSub Webhook时发生异常: {e}")
            return False
    
    def _get_app_access_token(self):
        """获取应用访问令牌"""
        try:
            url = "https://id.twitch.tv/oauth2/token"
            params = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'client_credentials'
            }
            
            response = requests.post(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info("成功获取应用访问令牌")
                return data['access_token']
            else:
                self.logger.error(f"获取应用访问令牌失败，状态码: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"获取应用访问令牌时发生异常: {e}")
            return None
    
    def _create_subscription_event(self, callback_url, app_access_token):
        """创建EventSub订阅事件"""
        try:
            url = "https://api.twitch.tv/helix/eventsub/subscriptions"
            headers = {
                'Client-ID': self.client_id,
                'Authorization': f'Bearer {app_access_token}',
                'Content-Type': 'application/json'
            }
            
            # 订阅事件数据
            payload = {
                "type": "channel.subscribe",
                "version": "1",
                "condition": {
                    "broadcaster_user_id": self.user_id
                },
                "transport": {
                    "method": "webhook",
                    "callback": callback_url,
                    "secret": self._generate_secret()
                }
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 202:
                self.logger.info("成功创建EventSub订阅事件")
                self.logger.info(f"回调URL: {callback_url}")
                self.logger.info("请注意：Twitch会发送一个验证请求到您的回调URL")
                return True
            else:
                self.logger.error(f"创建EventSub订阅事件失败，状态码: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
                
                # 提供一些常见错误的解决方案
                if response.status_code == 400:
                    self.logger.info("可能的解决方案：")
                    self.logger.info("1. 确保回调URL是公开可访问的，并且使用HTTPS")
                    self.logger.info("2. 确保回调URL的端口是80、443、8080或8443之一")
                    self.logger.info("3. 确保回调URL配置正确")
                return False
        except Exception as e:
            self.logger.error(f"创建EventSub订阅事件时发生异常: {e}")
            return False
    
    def _generate_secret(self):
        """生成一个随机密钥用于EventSub验证"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _start_eventsub_server(self, port):
        """启动本地HTTP服务器接收EventSub回调"""
        class EventSubHandler(http.server.BaseHTTPRequestHandler):
            """处理EventSub回调的请求处理器"""
            def do_GET(self):
                # 处理验证请求
                parsed_url = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed_url.query)
                
                if 'hub.challenge' in params:
                    # 这是一个验证请求，返回challenge值
                    challenge = params['hub.challenge'][0]
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(challenge.encode('utf-8'))
                    self.server.parent.logger.info("成功响应EventSub验证请求")
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def do_POST(self):
                # 处理事件通知
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    # 解析事件数据
                    event_data = json.loads(post_data.decode('utf-8'))
                    
                    # 处理订阅事件
                    if event_data.get('subscription', {}).get('type') == 'channel.subscribe':
                        event = event_data.get('event', {})
                        subscriber_name = event.get('user_name')
                        subscriber_id = event.get('user_id')
                        tier = event.get('tier')
                        message = event.get('message')
                        
                        # 将等级转换为可读格式
                        tier_map = {'1000': 'Tier 1', '2000': 'Tier 2', '3000': 'Tier 3'}
                        tier_readable = tier_map.get(tier, f'Unknown ({tier})')
                        
                        # 显示订阅信息
                        print("\n🎉 收到新订阅!")
                        print(f"  订阅者: {subscriber_name} (ID: {subscriber_id})")
                        print(f"  等级: {tier_readable}")
                        if message:
                            print(f"  订阅消息: {message}")
                        print("====================================")
                        
                        # 记录到日志
                        self.server.parent.logger.info(f"收到订阅: {subscriber_name} (ID: {subscriber_id}, 等级: {tier_readable})")
                        if message:
                            self.server.parent.logger.info(f"订阅消息: {message}")
                    
                    # 返回成功响应
                    self.send_response(200)
                    self.end_headers()
                    
                except Exception as e:
                    self.server.parent.logger.error(f"处理EventSub事件时发生异常: {e}")
                    self.send_response(500)
                    self.end_headers()
                    
            # 禁用日志输出
            def log_message(self, format, *args):
                return
        
        # 创建服务器实例
        server_address = ('', port)
        httpd = socketserver.TCPServer(server_address, EventSubHandler)
        httpd.parent = self  # 设置父引用，以便在处理器中访问
        
        self.logger.info(f"本地服务器已启动，监听端口: {port}")
        
        try:
            # 启动服务器
            httpd.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("正在关闭服务器...")
            httpd.shutdown()
            return True


def print_usage():
    """打印使用说明"""
    print("\n===== Twitch订阅消息读取工具 ======")
    print("此工具演示如何读取Twitch订阅的文字消息")
    print("提供两种方式:")
    print("1. IRC聊天方式 - 简单，直接从聊天中接收通知")
    print("2. EventSub Webhook方式 - 更强大，支持更多事件类型")
    print("\n使用方法:")
    print("  python twitch_sub_message_reader.py irc     # 使用IRC方式")
    print("  python twitch_sub_message_reader.py eventsub <callback_url> [port]  # 使用EventSub方式")
    print("\n环境变量设置:")
    print("- 所有方式都需要设置: TWITCH_ID, TWITCH_OAUTH_TOKEN")
    print("- EventSub方式还需要设置: TWITCH_SECRET")
    print("\n注意事项:")
    print("- EventSub的回调URL必须是公开可访问的HTTPS URL")
    print("- 本地测试时，可以使用ngrok等工具创建临时的公开URL")
    print("- 当有订阅发生时，工具会显示订阅者名称、等级和文字消息")


def main():
    """主函数"""
    print("\n🎉 Welcome to Twitch Subscription Message Reader! 🎉")
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print_usage()
        return
    
    # 获取模式参数
    mode = sys.argv[1].lower()
    
    # 初始化消息读取器
    reader = TwitchSubMessageReader()
    
    # 根据模式执行不同的操作
    if mode == 'irc':
        # 使用IRC方式
        reader.read_sub_messages_via_irc()
    elif mode == 'eventsub':
        # 使用EventSub方式
        # 获取回调URL和端口
        callback_url = sys.argv[2] if len(sys.argv) > 2 else None
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
        
        if not callback_url:
            print("错误: 使用EventSub方式需要提供回调URL")
            print_usage()
            return
        
        reader.read_sub_messages_via_eventsub(callback_url, port)
    else:
        print(f"错误: 未知的模式 '{mode}'")
        print_usage()
        return


if __name__ == '__main__':
    main()