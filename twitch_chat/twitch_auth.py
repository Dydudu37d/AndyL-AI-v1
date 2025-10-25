import requests
import os
import dotenv
import webbrowser
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# 加载环境变量
dotenv.load_dotenv()

# 全局变量用于存储授权码
authorization_code = None

class OAuthHandler(BaseHTTPRequestHandler):
    """处理OAuth回调的HTTP处理器"""
    def do_GET(self):
        global authorization_code
        
        # 解析URL获取查询参数
        parsed_url = urlparse(self.path)
        if parsed_url.path == '/callback':
            # 提取授权码
            query_params = parse_qs(parsed_url.query)
            if 'code' in query_params:
                authorization_code = query_params['code'][0]
                
                # 返回成功响应
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization successful!</h1><p>Please return to the command line to continue</p></body></html>")
            else:
                # 返回错误响应
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization failed!</h1><p>Authorization code not found.</p></body></html>")

class TwitchAuth:
    """Twitch API认证管理类"""
    
    def __init__(self):
        """初始化Twitch认证管理器"""
        self.client_id = os.getenv('TWITCH_ID')
        self.client_secret = os.getenv('TWITCH_SECRET')
        self.access_token = os.getenv('TWITCH_OAUTH_TOKEN') or os.getenv('TWITCH_ACCESS_TOKEN')
        self.refresh_token = os.getenv('TWITCH_REFRESH_TOKEN')
        self.redirect_uri = 'http://localhost:3009/callback'
        
        # 验证必要的环境变量
        if not self.client_id or not self.client_secret:
            raise ValueError("TWITCH_ID and TWITCH_SECRET must be set in .env file")
    
    def get_headers(self):
        """获取包含认证信息的请求头"""
        if not self.access_token:
            raise ValueError("No access token available. Please set TWITCH_OAUTH_TOKEN or TWITCH_ACCESS_TOKEN in .env file")
        
        return {
            'Client-ID': self.client_id,
            'Authorization': f'Bearer {self.access_token}'
        }
    
    def validate_token(self):
        """验证访问令牌是否有效"""
        url = "https://id.twitch.tv/oauth2/validate"
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                print("✓ Access token is valid")
                token_info = response.json()
                print(f"  User ID: {token_info.get('user_id')}")
                print(f"  User Name: {token_info.get('login')}")
                print(f"  Client ID: {token_info.get('client_id')}")
                print(f"  Scopes: {', '.join(token_info.get('scopes', []))}")
                return True, token_info
            else:
                print(f"✗ Access token is invalid or expired: {response.text}")
                return False, None
        except Exception as e:
            print(f"Error validating token: {e}")
            return False, None
    
    def refresh_access_token(self):
        """使用刷新令牌获取新的访问令牌"""
        if not self.refresh_token:
            raise ValueError("No refresh token available. Please set TWITCH_REFRESH_TOKEN in .env file")
        
        url = "https://id.twitch.tv/oauth2/token"
        params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token
        }
        
        try:
            response = requests.post(url, params=params)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                
                # 如果Twitch返回了新的刷新令牌，更新它
                if 'refresh_token' in token_data:
                    self.refresh_token = token_data['refresh_token']
                    print("New refresh token received")
                
                print("✓ Access token successfully refreshed")
                print(f"  New access token: {self.access_token}")
                print(f"  Expires in: {token_data.get('expires_in')} seconds")
                print(f"  Scope: {token_data.get('scope')}")
                
                # 注意：在实际应用中，你可能想要将新的令牌保存到.env文件或数据库中
                print("\nIMPORTANT: Update your .env file with the new tokens:")
                print(f"TWITCH_OAUTH_TOKEN={self.access_token}")
                print(f"TWITCH_REFRESH_TOKEN={self.refresh_token}")
                
                return True, token_data
            else:
                print(f"✗ Failed to refresh access token: {response.text}")
                return False, None
        except Exception as e:
            print(f"Error refreshing token: {e}")
            return False, None
    
    def get_user_info(self):
        """获取当前认证用户的信息"""
        url = "https://api.twitch.tv/helix/users"
        headers = self.get_headers()
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    user_info = data['data'][0]
                    print("\nUser Information:")
                    print(f"  Display Name: {user_info.get('display_name')}")
                    print(f"  User ID: {user_info.get('id')}")
                    print(f"  Login: {user_info.get('login')}")
                    print(f"  Description: {user_info.get('description')}")
                    print(f"  Profile Image URL: {user_info.get('profile_image_url')}")
                    return user_info
                else:
                    print("No user data returned")
                    return None
            else:
                print(f"Failed to get user info: {response.text}")
                return None
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None
    
    def get_app_access_token(self):
        """获取应用访问令牌（不需要用户认证）"""
        url = "https://id.twitch.tv/oauth2/token"
        params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(url, params=params)
            if response.status_code == 200:
                token_data = response.json()
                print("✓ App access token obtained")
                print(f"  Token: {token_data['access_token']}")
                print(f"  Expires in: {token_data.get('expires_in')} seconds")
                return token_data['access_token']
            else:
                raise Exception(f"Failed to get app access token: {response.text}")
        except Exception as e:
            print(f"Error getting app access token: {e}")
            return None
    
    def obtain_user_tokens(self):
        """使用OAuth授权码流程获取用户访问令牌和刷新令牌"""
        global authorization_code
        
        # 定义所需的权限范围
        scopes = [
            'user:read:email',
            'chat:read',
            'chat:edit',
            'channel:read:redemptions',
            'channel:moderate',
            'whispers:read',
            "channel:read:stream_key",#不要刪除！！！！！！！！！！！！！！！！！
            "channel:manage:broadcast",#不要刪除！！！！！！！！！！！！！！！！！
            "channel:manage:schedule",#不要刪除！！！！！！！！！！！！！！！！！
            "user:read:chat",#不要刪除！！！！！！！！！！！！！！！！！
            "user:write:chat"#不要刪除！！！！！！！！！！！！！！！！！
        ]
        
        # 构建授权URL
        auth_url = f"https://id.twitch.tv/oauth2/authorize?"
        auth_url += f"client_id={self.client_id}&"
        auth_url += f"redirect_uri={self.redirect_uri}&"
        auth_url += "response_type=code&"
        auth_url += f"scope={' '.join(scopes)}&"
        auth_url += "force_verify=true"
        
        print("\n===== 获取用户访问令牌和刷新令牌 =====")
        print("1. 请在浏览器中完成Twitch授权")
        print("2. 授权成功后，会自动跳转到本地服务器")
        print("3. 然后返回此命令行继续操作")
        print(f"\n授权URL: {auth_url}")
        
        # 自动打开浏览器
        webbrowser.open(auth_url)
        
        # 启动本地HTTP服务器接收授权码
        server_address = ('', 3009)  # 与redirect_uri中的端口保持一致
        httpd = HTTPServer(server_address, OAuthHandler)
        
        print("\n本地服务器已启动，等待授权回调...")
        
        # 等待授权码，超时时间为300秒（5分钟）
        timeout = time.time() + 300  # 5分钟后超时
        while authorization_code is None and time.time() < timeout:
            httpd.handle_request()
            time.sleep(1)
        
        if authorization_code is None:
            print("授权超时，请重试。")
            return False, None
        
        print(f"\n✓ 成功获取授权码: {authorization_code}")
        
        # 使用授权码获取访问令牌和刷新令牌
        token_url = "https://id.twitch.tv/oauth2/token"
        token_params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        try:
            response = requests.post(token_url, params=token_params)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data['refresh_token']
                
                print("\n✓ 成功获取访问令牌和刷新令牌！")
                print(f"  访问令牌: {self.access_token}")
                print(f"  刷新令牌: {self.refresh_token}")
                print(f"  过期时间: {token_data.get('expires_in')}秒")
                print(f"  权限范围: {', '.join(token_data.get('scope', []))}")
                
                print("\n请将以下内容添加到您的.env文件中：")
                print(f"TWITCH_OAUTH_TOKEN={self.access_token}")
                print(f"TWITCH_REFRESH_TOKEN={self.refresh_token}")
                
                return True, token_data
            else:
                print(f"✗ 无法获取令牌: HTTP状态码 {response.status_code}")
                print(f"✗ 响应内容: {response.text}")
                print("✗ 请检查您的应用程序设置和请求的权限范围")
                return False, None
        except Exception as e:
            print(f"获取令牌时出错: {e}")
            return False, None

def main():
    """主函数，演示Twitch认证功能"""
    print("===== Twitch Authentication Demo =====")
    print(" 注意：Twitchapps的TMI Token Generator已于2023年停止服务")
    print(" 本工具提供了替代方案来获取Twitch认证令牌")
    
    try:
        # 创建TwitchAuth实例
        twitch_auth = TwitchAuth()
        
        print(f"\n使用的Client ID: {twitch_auth.client_id}")
        
        # 显示菜单
        print("\n请选择要执行的操作：")
        print("1. 验证当前访问令牌")
        print("2. 刷新访问令牌")
        print("3. 获取用户信息")
        print("4. 获取应用访问令牌")
        print("5. 【新功能】通过OAuth授权码流程获取用户访问令牌和刷新令牌")
        
        choice = input("请输入选项编号 (1-5): ")
        
        if choice == '1':
            # 验证当前访问令牌
            is_valid, token_info = twitch_auth.validate_token()
            if is_valid and token_info:
                twitch_auth.get_user_info()
        elif choice == '2':
            # 刷新访问令牌
            if twitch_auth.refresh_token:
                print("\n正在尝试刷新访问令牌...")
                refreshed, new_token_data = twitch_auth.refresh_access_token()
                if refreshed:
                    twitch_auth.get_user_info()
            else:
                print("错误：未设置刷新令牌。请先使用选项5获取刷新令牌。")
        elif choice == '3':
            # 获取用户信息
            twitch_auth.get_user_info()
        elif choice == '4':
            # 获取应用访问令牌（不需要用户授权的令牌）
            print("\n获取应用访问令牌（用于服务器到服务器的API调用）...")
            app_token = twitch_auth.get_app_access_token()
        elif choice == '5':
            # 使用OAuth授权码流程获取用户访问令牌和刷新令牌
            print("\n此功能将引导您完成Twitch OAuth授权流程，获取访问令牌和刷新令牌")
            print("请确保您的Twitch开发者应用程序已正确配置，且重定向URI设置为：http://localhost:3009/callback")
            confirm = input("是否继续？(y/n): ")
            
            if confirm.lower() == 'y':
                success, token_data = twitch_auth.obtain_user_tokens()
                if success:
                    print("\n授权成功！您现在可以使用这些令牌来访问Twitch API和聊天室。")
        else:
            print("无效的选项，请重试。")
            
    except Exception as e:
        print(f"错误: {e}")
        print("\n如果您是开发者，请确保已在.env文件中设置了正确的TWITCH_ID和TWITCH_SECRET")
        print("如果您是普通用户，请参考twitch_auth_guide.md文件获取详细指导")

if __name__ == "__main__":
    main()