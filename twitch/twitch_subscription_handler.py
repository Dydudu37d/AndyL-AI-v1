import os
import sys
import socket
import threading
import time
import requests
import json
import logging
import http.server
import socketserver
import dotenv
import urllib.parse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitch_subscription.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TwitchSubscription")

class TwitchSubscriptionHandler:
    """处理Twitch订阅通知的类"""
    def __init__(self):
        # 加载环境变量
        dotenv.load_dotenv()
        
        # 获取配置
        self.client_id = os.getenv('TWITCH_ID')
        self.client_secret = os.getenv('TWITCH_SECRET')
        self.oauth_token = os.getenv('TWITCH_OAUTH_TOKEN')
        self.refresh_token = os.getenv('TWITCH_REFRESH_TOKEN')
        
        # 如果令牌以'oauth:'开头，移除这个前缀
        if self.oauth_token and self.oauth_token.startswith('oauth:'):
            self.oauth_token = self.oauth_token[6:]
        
        # 检查必要的配置
        self._check_config()
        
        # 用户信息
        self.user_id = None
        self.user_name = None
        
        # 获取用户信息
        self._get_user_info()
        
    def _check_config(self):
        """检查必要的配置是否存在"""
        required_vars = ['TWITCH_ID', 'TWITCH_OAUTH_TOKEN']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"缺少必要的环境变量: {', '.join(missing_vars)}")
            logger.error("请在.env文件中添加这些变量")
            sys.exit(1)
        
        logger.info("配置检查完成，所有必要的环境变量都已设置")
    
    def _get_user_info(self):
        """获取当前用户的信息"""
        try:
            url = "https://api.twitch.tv/helix/users"
            headers = {
                'Client-ID': self.client_id,
                'Authorization': f'Bearer {self.oauth_token}'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    self.user_id = data['data'][0]['id']
                    self.user_name = data['data'][0]['login']
                    logger.info(f"成功获取用户信息: {self.user_name} (ID: {self.user_id})")
                else:
                    logger.error("未能获取用户信息，响应中没有数据")
            else:
                logger.error(f"获取用户信息失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
        except Exception as e:
            logger.error(f"获取用户信息时发生异常: {e}")
    
    def connect_via_irc(self, port=6667):
        """通过IRC方式连接到Twitch聊天，接收订阅通知"""
        logger.info("正在通过IRC方式连接到Twitch聊天...")
        
        # 确保用户信息已获取
        if not self.user_id or not self.user_name:
            logger.error("无法连接到IRC，因为未能获取用户信息")
            return False
        
        try:
            # 创建IRC连接
            server = 'irc.chat.twitch.tv'
            nick = 'justinfan12345'  # 使用默认的匿名用户名
            
            # 连接到服务器
            irc = socket.socket()
            irc.connect((server, port))
            
            # 发送认证信息
            irc.send(f"PASS oauth:{self.oauth_token}\r\n".encode('utf-8'))
            irc.send(f"NICK {nick}\r\n".encode('utf-8'))
            irc.send(f"JOIN #{self.user_name}\r\n".encode('utf-8'))
            
            logger.info(f"已连接到 {self.user_name} 的聊天室")
            logger.info("正在监听订阅通知... (按Ctrl+C退出)")
            
            # 监听聊天消息
            while True:
                # 接收数据
                data = irc.recv(2048).decode('utf-8')
                
                # 回复PING请求以保持连接
                if data.startswith('PING'):
                    irc.send("PONG\r\n".encode('utf-8'))
                
                # 处理订阅通知
                if 'USERNOTICE' in data and 'subscriber' in data:
                    self._handle_subscription_notice(data)
                
                # 输出接收到的数据（调试用）
                # logger.debug(f"接收到数据: {data}")
                
        except KeyboardInterrupt:
            logger.info("正在退出IRC连接...")
            return True
        except Exception as e:
            logger.error(f"IRC连接发生异常: {e}")
            return False
    
    def _handle_subscription_notice(self, data):
        """处理IRC订阅通知"""
        try:
            # 解析订阅信息
            # 这是一个简化的解析，实际的USERNOTICE消息格式可能更复杂
            parts = data.split(';')
            
            # 获取订阅者名称
            subscriber_name = None
            for part in parts:
                if part.startswith('display-name='):
                    subscriber_name = part.split('=')[1]
                    break
            
            # 获取订阅消息
            message = None
            if ':' in data:
                message_parts = data.split(':', 2)
                if len(message_parts) > 2:
                    message = message_parts[2].strip()
            
            # 记录订阅信息
            logger.info(f"🎉 收到新订阅！订阅者: {subscriber_name}")
            if message:
                logger.info(f"💬 订阅消息: {message}")
            
            # 这里可以添加自定义的订阅处理逻辑，比如发送感谢消息等
            self._on_new_subscription(subscriber_name, message)
            
        except Exception as e:
            logger.error(f"处理订阅通知时发生异常: {e}")
    
    def _on_new_subscription(self, subscriber_name, message):
        """当收到新订阅时的回调方法，可在子类中重写"""
        # 示例：发送感谢消息到聊天室
        # 注意：要发送消息，需要额外的权限和代码
        pass
    
    def setup_eventsub_webhook(self, callback_url, server_port=8080):
        """设置EventSub Webhook接收订阅通知"""
        logger.info("正在设置EventSub Webhook...")
        
        # 确保用户信息已获取
        if not self.user_id or not self.user_name:
            logger.error("无法设置EventSub Webhook，因为未能获取用户信息")
            return False
        
        try:
            # 1. 获取应用访问令牌（用于创建EventSub订阅）
            app_access_token = self._get_app_access_token()
            if not app_access_token:
                logger.error("无法获取应用访问令牌，无法设置EventSub Webhook")
                return False
            
            # 2. 创建订阅事件
            subscription_success = self._create_subscription_event(callback_url, app_access_token)
            if not subscription_success:
                logger.error("创建订阅事件失败")
                return False
            
            # 3. 启动本地服务器接收回调
            logger.info(f"正在启动本地服务器，监听端口: {server_port}")
            self._start_local_server(server_port)
            
        except Exception as e:
            logger.error(f"设置EventSub Webhook时发生异常: {e}")
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
                logger.info("成功获取应用访问令牌")
                return data['access_token']
            else:
                logger.error(f"获取应用访问令牌失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return None
        except Exception as e:
            logger.error(f"获取应用访问令牌时发生异常: {e}")
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
                    "secret": self._generate_secret()  # 生成一个随机密钥用于验证
                }
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 202:
                logger.info("成功创建EventSub订阅事件")
                logger.info(f"回调URL: {callback_url}")
                logger.info("请注意：Twitch会发送一个验证请求到您的回调URL")
                return True
            else:
                logger.error(f"创建EventSub订阅事件失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                
                # 提供一些常见错误的解决方案
                if response.status_code == 400:
                    logger.info("可能的解决方案：")
                    logger.info("1. 确保回调URL是公开可访问的，并且使用HTTPS")
                    logger.info("2. 确保回调URL的端口是80、443、8080或8443之一")
                    logger.info("3. 确保回调URL配置正确")
                return False
        except Exception as e:
            logger.error(f"创建EventSub订阅事件时发生异常: {e}")
            return False
    
    def _generate_secret(self):
        """生成一个随机密钥用于EventSub验证"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _start_local_server(self, port):
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
                    logger.info("成功响应EventSub验证请求")
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
                    logger.info(f"接收到EventSub事件: {event_data}")
                    
                    # 处理订阅事件
                    if event_data.get('subscription', {}).get('type') == 'channel.subscribe':
                        event = event_data.get('event', {})
                        subscriber_name = event.get('user_name')
                        tier = event.get('tier')
                        message = event.get('message')
                        
                        logger.info(f"🎉 收到新订阅！订阅者: {subscriber_name}, 等级: {tier}")
                        if message:
                            logger.info(f"💬 订阅消息: {message}")
                        
                        # 调用自定义的订阅处理方法
                        self.server.parent._on_new_subscription(subscriber_name, message)
                    
                    # 返回成功响应
                    self.send_response(200)
                    self.end_headers()
                    
                except Exception as e:
                    logger.error(f"处理EventSub事件时发生异常: {e}")
                    self.send_response(500)
                    self.end_headers()
                    
            # 禁用日志输出
            def log_message(self, format, *args):
                return
        
        # 创建服务器实例
        server_address = ('', port)
        httpd = socketserver.TCPServer(server_address, EventSubHandler)
        httpd.parent = self  # 设置父引用，以便在处理器中访问
        
        logger.info(f"本地服务器已启动，监听端口: {port}")
        logger.info("等待接收EventSub事件通知... (按Ctrl+C退出)")
        
        try:
            # 启动服务器
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("正在关闭服务器...")
            httpd.shutdown()
            return True

class CustomSubscriptionHandler(TwitchSubscriptionHandler):
    """自定义订阅处理器，重写_on_new_subscription方法实现自定义逻辑"""
    def _on_new_subscription(self, subscriber_name, message):
        """自定义订阅处理逻辑"""
        # 这里可以添加你想要的任何订阅处理逻辑
        # 例如：发送感谢消息、记录到数据库、触发其他事件等
        logger.info(f"Processing subscription for {subscriber_name}...")
        
        # 示例：输出感谢消息
        if subscriber_name:
            print(f"\nThank you so much {subscriber_name} for subscribing! ❤️❤️❤️")
            if message:
                print(f"💬 {subscriber_name} says: {message}")

def print_usage():
    """打印使用说明"""
    print("\n===== Twitch Subscription Tool =====")
    print("This tool provides two ways to receive Twitch subscription notifications:")
    print("1. IRC Chat Method - Simple, receive notifications directly from chat")
    print("2. EventSub Webhook Method - More powerful, supports more event types")
    print("\nUsage:")
    print("  python twitch_subscription_handler.py irc     # Use IRC method")
    print("  python twitch_subscription_handler.py eventsub [callback_url] [port]  # Use EventSub method")
    print("\nNotes:")
    print("- For IRC method, set TWITCH_ID and TWITCH_OAUTH_TOKEN in .env file")
    print("- For EventSub method, also set TWITCH_SECRET, and callback URL must be publicly accessible HTTPS URL")
    print("- For local testing of EventSub, you can use tools like ngrok to create temporary public URL")

def main():
    """主函数"""
    # 打印欢迎信息
    print("\n🎉 Welcome to Twitch Subscription Notifier! 🎉")
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print_usage()
        return
    
    # 获取模式参数
    mode = sys.argv[1].lower()
    
    # 初始化处理器
    handler = CustomSubscriptionHandler()
    
    # 根据模式执行不同的操作
    if mode == 'irc':
        # 使用IRC方式
        handler.connect_via_irc()
    elif mode == 'eventsub':
        # 使用EventSub方式
        # 获取回调URL和端口
        callback_url = sys.argv[2] if len(sys.argv) > 2 else None
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
        
        if not callback_url:
            print("错误: 使用EventSub方式需要提供回调URL")
            print_usage()
            return
        
        handler.setup_eventsub_webhook(callback_url, port)
    else:
        print(f"错误: 未知的模式 '{mode}'")
        print_usage()
        return

if __name__ == '__main__':
    main()