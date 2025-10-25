import os
import webbrowser
import http.server
import socketserver
import json
import time
import dotenv
import requests

# 加载环境变量
dotenv.load_dotenv()

# 全局变量
auth_code = None

def is_port_in_use(port):
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class OAuthHandler(http.server.BaseHTTPRequestHandler):
    """处理OAuth回调的请求处理器"""
    def do_GET(self):
        global auth_code
        
        print(f"收到回调请求: {self.path}")
        
        # 解析URL，提取授权码
        if self.path.startswith('/callback') or 'code=' in self.path or 'error=' in self.path:
            # 提取授权码
            query = self.path.split('?', 1)[1] if '?' in self.path else ''
            params = dict(pair.split('=') for pair in query.split('&') if '=' in pair)
            
            if 'code' in params:
                auth_code = params['code']
                print(f"成功获取授权码: {auth_code[:10]}...")
                # 发送成功响应
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write('<html><body><h1>授权成功！</h1><p>您可以关闭此窗口并返回终端。</p></body></html>'.encode('utf-8'))
            else:
                # 处理错误情况
                error_msg = params.get('error_description', '未知错误')
                error_type = params.get('error', 'unknown_error')
                print(f"授权失败: {error_type} - {error_msg}")
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f'<html><body><h1>授权失败</h1><p>错误类型: {error_type}</p><p>错误描述: {error_msg}</p></body></html>'.encode('utf-8'))
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
        self.redirect_uri = 'http://localhost:3009/callback'  # 确保与Twitch开发者控制台中配置的一致
        self.server_port = 3009  # 确保端口与重定向URI一致
        
        # 验证必要的环境变量是否存在
        if not self.client_id or not self.client_secret:
            print("错误: 请确保在.env文件中设置了TWITCH_ID和TWITCH_SECRET")
            exit(1)
            
        # 检查端口是否被占用
        if is_port_in_use(self.server_port):
            print(f"错误: 端口 {self.server_port} 已被占用，请关闭占用该端口的程序或修改端口号")
            exit(1)
            
    def obtain_user_tokens(self):
        """获取用户授权并交换访问令牌"""
        # 定义请求的权限范围 - 使用基础scope组合
        scopes = [
            'user:read:email',
            'chat:read',
            'chat:edit',
            'channel:read:redemptions',
            'channel:moderate',
            'user:read:chat',
            'user:write:chat'
            # 移除可能导致问题的scope
        ]
        
        # 构建授权URL - 注意正确的格式
        auth_url = (
            f"https://id.twitch.tv/oauth2/authorize?"
            f"client_id={self.client_id}&"
            f"redirect_uri={self.redirect_uri}&"
            f"response_type=code&"
            f"scope={'%20'.join(scopes)}&"
            f"force_verify=true"
        )
        
        print(f"\n请在浏览器中授权以下权限: {'、'.join(scopes)}")
        print(f"授权URL: {auth_url}")
        
        # 启动本地HTTP服务器等待回调
        with socketserver.TCPServer(('', self.server_port), OAuthHandler) as httpd:
            print(f"本地服务器已启动在端口 {self.server_port}，等待回调...")
            
            # 自动打开浏览器
            try:
                webbrowser.open(auth_url)
                print("已尝试自动打开浏览器进行授权")
            except Exception as e:
                print(f"警告: 无法自动打开浏览器，请手动复制上面的URL到浏览器中。错误: {e}")
            
            # 等待授权码，超时时间设为3分钟
            start_time = time.time()
            timeout = 180  # 3分钟
            
            while auth_code is None:
                httpd.handle_request()
                if time.time() - start_time > timeout:
                    print("错误: 授权超时，请重新运行脚本并在3分钟内完成授权")
                    return False, None
            
            # 使用授权码交换访问令牌
            token_url = 'https://id.twitch.tv/oauth2/token'
            token_params = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': auth_code,
                'grant_type': 'authorization_code',
                'redirect_uri': self.redirect_uri
            }
            
            print("正在交换访问令牌...")
            try:
                response = requests.post(token_url, params=token_params)
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data.get('access_token')
                    self.refresh_token = token_data.get('refresh_token')
                    print("成功获取访问令牌和刷新令牌！")
                    print(f"令牌有效期: {token_data.get('expires_in')}秒")
                    print(f"获得的权限范围: {token_data.get('scope')}")
                    return True, token_data
                else:
                    print(f"错误: 获取访问令牌失败，状态码: {response.status_code}")
                    print(f"响应内容: {response.text}")
                    print("可能的原因：")
                    print("1. 重定向URI与Twitch开发者控制台中配置的不匹配")
                    print("2. 授权码已过期或无效")
                    print("3. 客户端ID或客户端密钥错误")
                    return False, None
            except Exception as e:
                print(f"错误: 交换令牌过程中出现异常: {e}")
                return False, None
    
    def validate_token(self):
        """验证访问令牌的有效性"""
        if not self.access_token:
            print("错误: 没有可用的访问令牌进行验证")
            return False, None
            
        validate_url = 'https://id.twitch.tv/oauth2/validate'
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
    
    def write_tokens_to_env(self, token_data):
        """将令牌写入.env文件"""
        try:
            # 读取现有.env文件
            with open('.env', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 更新或添加令牌行
            token_lines = {
                'TWITCH_ACCESS_TOKEN': token_data.get('access_token'),
                'TWITCH_REFRESH_TOKEN': token_data.get('refresh_token')
            }
            
            # 检查并替换现有行
            for key, value in token_lines.items():
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith(f'{key}='):
                        lines[i] = f'{key}={value}\n'
                        updated = True
                        break
                if not updated:
                    lines.append(f'{key}={value}\n')
                    
            # 写回.env文件
            with open('.env', 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
            print("已将令牌写入.env文件")
            return True
        except Exception as e:
            print(f"错误: 无法写入.env文件: {e}")
            return False

def test_scope_validity():
    """测试工具：帮助诊断scope有效性问题"""
    print("\n===== Twitch Scope诊断工具 ====")
    print("此工具可以帮助您找出可能导致'无效scope'错误的问题")
    print("\n建议的排查步骤：")
    print("1. 尝试使用简化的scope组合进行授权")
    print("2. 如果简化组合成功，逐步添加其他scope")
    print("3. 如果遇到错误，最近添加的scope很可能是问题所在")
    print("4. 请参考Twitch最新的API文档，确保使用的scope仍然受支持")
    print("\n关于redirect_mismatch错误的提示：")
    print("- 确保Twitch开发者控制台中注册的重定向URI与代码中的完全一致")
    print("- 包括协议(http/https)、主机名(localhost)、端口号和路径")
    print("- 当前代码中使用的重定向URI是: http://localhost:3009/callback")
    
    # 提供一个测试不同scope组合的简单方法
    print("\n以下是一些可能的测试组合:")
    print("\n组合A (最基础的scope):")
    print("user:read:email chat:read chat:edit")
    
    print("\n组合B (增加一些常用scope):")
    print("user:read:email chat:read chat:edit channel:read:redemptions channel:moderate")
    
    print("\n要使用特定的scope组合，请修改代码中的scopes列表")

def main():
    """主函数"""
    print("===== Twitch OAuth认证工具 ======")
    print(f"使用的客户端ID: {os.getenv('TWITCH_ID')}")
    print(f"使用的重定向URI: {os.getenv('TWITCH_REDIRECT_URI', 'http://localhost:3009/callback')}")
    print(f"HTTP服务器端口: 3009")
    
    # 确保requests库已安装
    try:
        import requests
    except ImportError:
        print("错误: requests库未安装，请运行 'pip install requests' 安装")
        exit(1)
    
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
        print("\n重要提示：")
        print("1. 请确保Twitch开发者控制台中注册的重定向URI与当前代码中使用的完全一致")
        print("2. 当前代码中使用的重定向URI是: http://localhost:3009/callback")
        print("3. 如果需要使用不同的端口，请同时修改代码中的重定向URI和server_port")

if __name__ == '__main__':
    main()