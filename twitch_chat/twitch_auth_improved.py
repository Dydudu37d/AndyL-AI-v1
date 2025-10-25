import requests
import os
import webbrowser
import http.server
import socketserver
import json
import time
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 全局变量
auth_code = None

class OAuthHandler(http.server.BaseHTTPRequestHandler):
    """处理OAuth回调的请求处理器"""
    def do_GET(self):
        global auth_code
        
        # 解析URL，提取授权码
        if self.path.startswith('/callback'):
            # 提取授权码
            query = self.path.split('?', 1)[1] if '?' in self.path else ''
            params = dict(pair.split('=') for pair in query.split('&') if '=' in pair)
            
            if 'code' in params:
                auth_code = params['code']
                # 发送成功响应
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write('<html><body><h1>授权成功！</h1><p>您可以关闭此窗口并返回终端。</p></body></html>'.encode('utf-8'))
            else:
                # 处理错误情况
                error_msg = params.get('error_description', '未知错误')
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f'<html><body><h1>授权失败</h1><p>{error_msg}</p></body></html>'.encode('utf-8'))
        else:
            # 处理其他路径
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><body><h1>404 Not Found</h1></body></html>'.encode('utf-8'))

    # 禁用日志输出
    def log_message(self, format, *args):
        return

class TwitchAuth:
    """处理Twitch OAuth认证的类"""
    def __init__(self):
        # 从环境变量获取认证信息
        self.client_id = os.getenv('TWITCH_ID')
        self.client_secret = os.getenv('TWITCH_SECRET')
        self.access_token = None
        self.refresh_token = None
        self.redirect_uri = 'http://localhost:3009/callback'  # 与Twitch开发者控制台中配置的一致
        self.server_port = 3009  # 确保端口与重定向URI一致
        
        # 验证必要的环境变量是否存在
        if not self.client_id or not self.client_secret:
            print("错误: 请确保在.env文件中设置了TWITCH_ID和TWITCH_SECRET")
            exit(1)
            
    def obtain_user_tokens(self):
        """获取用户授权并交换访问令牌"""
        # 定义请求的权限范围
        scopes = [
            'user:read:email',
            'chat:read',
            'chat:edit',
            'channel:read:redemptions',
            'channel:moderate',
            'whispers:read',
            'channel:read:stream_key',  # 不要删除！
            'channel:manage:broadcast',  # 不要删除！
            'channel:manage:schedule',   # 不要删除！
            'user:read:chat',
            'user:write:chat'
        ]
        
        # 构建授权URL
        auth_url = (
            f"https://id.twitch.tv/oauth2/authorize?" \
            f"client_id={self.client_id}&" \
            f"redirect_uri={self.redirect_uri}&" \
            f"response_type=code&" \
            f"scope={' '.join(scopes)}&" \
            f"force_verify=true"
        )
        
        print(f"\n请在浏览器中授权以下权限: {'、'.join(scopes)}")
        print(f"授权URL: {auth_url}")
        
        # 自动打开浏览器
        try:
            webbrowser.open(auth_url)
            print("已尝试自动打开浏览器进行授权...")
        except Exception as e:
            print(f"无法自动打开浏览器: {e}")
            print("请手动复制上面的URL到浏览器中进行授权")
        
        # 启动本地HTTP服务器接收回调
        print(f"\n本地服务器已启动，等待授权回调...")
        with socketserver.TCPServer(('', self.server_port), OAuthHandler) as httpd:
            # 设置超时（60秒），避免无限等待
            httpd.timeout = 60
            try:
                httpd.handle_request()
            except KeyboardInterrupt:
                print("\n服务器已停止")
                return False, None
        
        # 检查是否获取到授权码
        if not auth_code:
            print("错误: 未获取到授权码")
            print("请检查是否在浏览器中完成了授权，或授权过程中出现了错误")
            return False, None
        
        print(f"成功获取授权码: {auth_code}")
        
        # 交换授权码获取访问令牌
        token_url = "https://id.twitch.tv/oauth2/token"
        token_params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': auth_code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        print("\n正在交换授权码获取访问令牌...")
        try:
            response = requests.post(token_url, params=token_params)
            print(f"HTTP状态码: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                
                print("\n成功获取令牌！")
                print(f"访问令牌: {self.access_token}")
                print(f"刷新令牌: {self.refresh_token}")
                print(f"令牌类型: {token_data.get('token_type')}")
                print(f"过期时间: {token_data.get('expires_in')}秒")
                print(f"已授权的权限范围: {', '.join(token_data.get('scope', []))}")
                
                # 检查是否有无效的scope导致的问题
                if 'scope' in token_data and token_data['scope']:
                    requested_scope_set = set(scopes)
                    granted_scope_set = set(token_data['scope'])
                    
                    if not requested_scope_set.issubset(granted_scope_set):
                        missing_scopes = requested_scope_set - granted_scope_set
                        print(f"警告: 以下请求的权限未被授予: {', '.join(missing_scopes)}")
                
                return True, token_data
            else:
                print(f"错误: 无法获取访问令牌")
                print(f"响应内容: {response.text}")
                
                # 特别处理scope错误
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        if 'message' in error_data and 'invalid scope' in error_data['message']:
                            print("\n诊断信息:")
                            print("1. 错误信息表明请求了无效的scope")
                            print("2. 请检查并移除无效的scope，或更新到Twitch当前支持的scope")
                            print("3. 建议使用提供的twitch_oauth_test.py脚本测试不同的scope组合")
                    except:
                        pass
                
                return False, None
        except Exception as e:
            print(f"错误: 请求过程中出现异常: {e}")
            return False, None
    
    def write_tokens_to_env(self, token_data):
        """将令牌信息写入.env文件"""
        if not token_data:
            return False
            
        # 读取现有的.env文件内容
        env_content = {}
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        env_content[key.strip()] = value.strip()
        
        # 更新令牌信息
        env_content['TWITCH_ACCESS_TOKEN'] = token_data.get('access_token', '')
        env_content['TWITCH_REFRESH_TOKEN'] = token_data.get('refresh_token', '')
        
        # 写回.env文件
        try:
            with open('.env', 'w', encoding='utf-8') as f:
                for key, value in env_content.items():
                    f.write(f"{key}={value}\n")
            print("\n令牌信息已成功写入.env文件")
            return True
        except Exception as e:
            print(f"错误: 无法写入.env文件: {e}")
            return False

    def refresh_access_token(self):
        """使用刷新令牌获取新的访问令牌"""
        if not self.refresh_token:
            print("错误: 没有可用的刷新令牌")
            return False
        
        token_url = "https://id.twitch.tv/oauth2/token"
        token_params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        try:
            response = requests.post(token_url, params=token_params)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')  # 可能会更新
                
                print("成功刷新访问令牌！")
                return True, token_data
            else:
                print(f"错误: 无法刷新访问令牌，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False, None
        except Exception as e:
            print(f"错误: 刷新令牌过程中出现异常: {e}")
            return False, None

    def validate_token(self):
        """验证访问令牌是否有效"""
        if not self.access_token:
            print("错误: 没有可用的访问令牌")
            return False
        
        validate_url = "https://id.twitch.tv/oauth2/validate"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        try:
            response = requests.get(validate_url, headers=headers)
            
            if response.status_code == 200:
                print("访问令牌有效！")
                validate_data = response.json()
                print(f"用户ID: {validate_data.get('user_id')}")
                print(f"用户名: {validate_data.get('login')}")
                print(f"客户端ID: {validate_data.get('client_id')}")
                return True, validate_data
            else:
                print(f"错误: 访问令牌无效，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False, None
        except Exception as e:
            print(f"错误: 验证令牌过程中出现异常: {e}")
            return False, None

def test_scope_validity():
    """测试工具：帮助诊断scope有效性问题"""
    print("\n===== Twitch Scope诊断工具 ====")
    print("此工具可以帮助您找出可能导致'无效scope'错误的问题")
    print("\n建议的排查步骤：")
    print("1. 尝试使用简化的scope组合进行授权")
    print("2. 如果简化组合成功，逐步添加其他scope")
    print("3. 如果遇到错误，最近添加的scope很可能是问题所在")
    print("4. 请参考Twitch最新的API文档，确保使用的scope仍然受支持")
    
    # 提供一个测试不同scope组合的简单方法
    print("\n以下是一些可能的测试组合:")
    print("\n组合A (最基础的scope):")
    print("user:read:email chat:read chat:edit")
    
    print("\n组合B (增加一些常用scope):")
    print("user:read:email chat:read chat:edit channel:read:redemptions channel:moderate")
    
    print("\n要使用特定的scope组合，请修改代码中的scopes列表")

def main():
    """主函数"""
    print("===== Twitch OAuth认证工具 =====")
    print(f"使用的客户端ID: {os.getenv('TWITCH_ID')}")
    print(f"使用的重定向URI: http://localhost:3009/callback")
    print(f"HTTP服务器端口: 3009")
    
    twitch_auth = TwitchAuth()
    
    # 首先显示scope诊断信息
    test_scope_validity()
    
    # 尝试获取用户令牌
    print("\n开始获取用户授权...")
    success, token_data = twitch_auth.obtain_user_tokens()
    
    if success and token_data:
        # 验证获取的令牌
        print("\n验证获取的令牌...")
        validate_success, _ = twitch_auth.validate_token()
        
        if validate_success:
            # 写入.env文件
            twitch_auth.write_tokens_to_env(token_data)
            print("\n认证流程已完成！")
        else:
            print("\n警告: 令牌验证失败，但可能是临时问题")
            print("已尝试将令牌写入.env文件，请稍后手动验证")
            twitch_auth.write_tokens_to_env(token_data)
    else:
        print("\n认证流程未完成！")
        print("请检查错误信息并尝试解决问题")
        print("建议使用提供的twitch_oauth_test.py脚本进行进一步诊断")

if __name__ == '__main__':
    main()