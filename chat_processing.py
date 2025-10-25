import requests
import dotenv
import os
import socket
import threading
import time
import re
import requests

# 加载环境变量
dotenv.load_dotenv()

msg = ""

# 获取应用访问令牌
def get_app_access_token(client_id, client_secret):
    """Get app access token (doesn't require user authentication)"""
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }
    
    response = requests.post(url, params=params)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Failed to get access token: {response.text}")

# 连接到Twitch IRC并读取聊天消息
def connect_to_twitch_chat(username, oauth_token, channel):
    """连接到Twitch聊天室并读取消息"""
    global msg  # 声明使用全局变量
    # Twitch IRC服务器信息
    server = 'irc.chat.twitch.tv'
    port = 6667
    
    try:
        # 创建socket连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server, port))
        
        # 发送认证信息
        sock.send(f"PASS {oauth_token}\r\n".encode('utf-8'))
        sock.send(f"NICK {username}\r\n".encode('utf-8'))
        sock.send(f"JOIN #{channel}\r\n".encode('utf-8'))
        
        print(f"已连接到 {channel} 的聊天室")
        print("正在监听聊天消息... (按Ctrl+C退出)")
        
        # 持续监听消息
        while True:
            response = sock.recv(2048).decode('utf-8')
            
            # 响应PONG以保持连接
            if response.startswith('PING'):
                sock.send("PONG :tmi.twitch.tv\r\n".encode('utf-8'))
            
            # 解析并显示聊天消息
            if 'PRIVMSG' in response:
                # 使用正则表达式提取用户名和消息
                match = re.search(r':(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)', response)
                if match:
                    username = match.group(1)
                    message = match.group(2)
                    print(f"[{username}]: {message}")
                    # 处理消息
                    msg = f"聊天室: [{username}]: {message}"
                    
            
            # 防止CPU占用过高
            time.sleep(0.5)
            
    except Exception as e:
        print(f"连接聊天室时出错: {e}")
    finally:
        if 'sock' in locals():
            sock.close()

def get_user_channel_info(client_id, access_token):
    """获取用户的频道信息"""
    # 首先获取当前认证用户的ID
    users_url = "https://api.twitch.tv/helix/users"
    headers = {
        'Client-ID': client_id,
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        # 获取认证用户信息
        users_response = requests.get(users_url, headers=headers)
        if users_response.status_code == 200:
            users_data = users_response.json()
            if users_data['data']:
                user = users_data['data'][0]
                user_id = user['id']
                user_name = user['login']
                
                # 获取频道信息
                channel_url = f"https://api.twitch.tv/helix/channels?broadcaster_id={user_id}"
                channel_response = requests.get(channel_url, headers=headers)
                
                if channel_response.status_code == 200:
                    channel_data = channel_response.json()
                    if channel_data['data']:
                        channel = channel_data['data'][0]
                        print(f"\n账号信息:")
                        print(f"用户ID: {user_id}")
                        print(f"用户名: {user_name}")
                        print(f"频道ID: {channel['broadcaster_id']}")
                        print(f"频道名称: {channel['broadcaster_name']}")
                        print(f"频道标题: {channel['title']}")
                        print(f"游戏分类: {channel['game_name']}")
                        return user_name  # 返回频道名称用于连接聊天室
            
        print(f"获取频道信息失败: {users_response.text}")
        return None
    except Exception as e:
        print(f"获取频道信息时出错: {e}")
        return None

def main():
    client_id = os.getenv('TWITCH_ID')
    client_secret = os.getenv('TWITCH_SECRET')
    twitch_oauth_token = os.getenv('TWITCH_OAUTH_TOKEN')  # 需要用户OAuth令牌用于聊天访问
    twitch_refresh_token = os.getenv('TWITCH_REFRESH_TOKEN')
    
    if not client_id or not client_secret:
        print("Error: TWITCH_ID and TWITCH_SECRET must be set in .env file")
        return
    
    # 如果没有提供用户OAuth令牌，提示用户
    if not twitch_oauth_token:
        print("警告: 未在.env文件中找到TWITCH_OAUTH_TOKEN")
        print("\n注意：Twitchapps的TMI Token Generator已于2023年停止服务！")
        print("\n您可以通过以下方式获取OAuth令牌：")
        print("1. 对于开发者：运行 python twitch_auth.py 并选择选项5")
        print("   这将引导您完成OAuth授权流程，获取您自己的访问令牌和刷新令牌")
        print("2. 对于普通用户：使用可靠的第三方令牌生成器")
        print("\n获取令牌后，请添加到.env文件中：")
        print("格式: TWITCH_OAUTH_TOKEN=oauth:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("\n详细指南请参考 twitch_auth_guide.md 文件")
        return
    
    try:
        # 使用用户OAuth令牌获取频道信息
        channel_name = get_user_channel_info(client_id, twitch_oauth_token)
        
        if channel_name:
            # 创建并启动聊天监听线程
            chat_thread = threading.Thread(target=connect_to_twitch_chat,
                                           args=('justinfan12345', twitch_oauth_token, channel_name))
            chat_thread.daemon = True
            chat_thread.start()
            
            # 主线程等待用户输入退出
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n正在退出...")
        else:
            print("无法获取频道信息，无法连接到聊天室")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()