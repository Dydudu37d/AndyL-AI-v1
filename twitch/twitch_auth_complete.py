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
        
        print(f"\n===== 收到回调请求 =====")
        print(f"完整路径: {self.path}")
        
        # 解析URL，提取授权码或错误信息
        query = self.path.split('?', 1)[1] if '?' in self.path else ''
        params = dict(pair.split('=') for pair in query.split('&') if '=' in pair)
        
        if 'code' in params:
            auth_code = params['code']
            print(f"✅ 成功获取授权码: {auth_code[:10]}...")
            # 发送成功响应
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('''<html><body>
                <h1>🎉 授权成功！</h1>
                <p>您已成功授权应用程序访问您的Twitch账号。</p>
                <p>请关闭此窗口并返回终端查看认证结果。</p>
                </body></html>'''.encode('utf-8'))
        else:
            # 处理错误情况
            error_msg = params.get('error_description', '未知错误')
            error_type = params.get('error', 'unknown_error')
            print(f"❌ 授权失败: {error_type} - {error_msg}")
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f'''<html><body>
                <h1>授权失败</h1>
                <p><strong>错误类型:</strong> {error_type}</p>
                <p><strong>错误描述:</strong> {error_msg}</p>
                <p>请返回终端查看详细的故障排除建议。</p>
                </body></html>'''.encode('utf-8'))

    # 禁用日志输出
    def log_message(self, format, *args):
        return

class TwitchAuth:
    """处理Twitch OAuth认证的类"""
    def __init__(self, redirect_uri, port):
        # 从环境变量获取认证信息
        self.client_id = os.getenv('TWITCH_ID')
        self.client_secret = os.getenv('TWITCH_SECRET')
        self.access_token = None
        self.refresh_token = None
        self.redirect_uri = redirect_uri
        self.server_port = port
        
        # 验证必要的环境变量是否存在
        if not self.client_id:
            print("❌ 错误: 请确保在.env文件中设置了TWITCH_ID")
            exit(1)
        
        if not self.client_secret:
            print("❌ 错误: 请确保在.env文件中设置了TWITCH_SECRET")
            exit(1)
            
        # 检查端口是否被占用
        if is_port_in_use(self.server_port):
            print(f"❌ 错误: 端口 {self.server_port} 已被占用")
            print("请关闭占用该端口的程序或修改端口号")
            exit(1)
            
    def obtain_user_tokens(self):
        """获取用户授权并交换访问令牌"""
        # 定义请求的权限范围 - 使用最基础的scope以确保兼容性
        scopes = [
            'user:read:email',
            'chat:read',
            'chat:edit'
        ]
        
        # 构建授权URL - 确保格式正确
        auth_url = (
            f"https://id.twitch.tv/oauth2/authorize?"
            f"client_id={self.client_id}&"
            f"redirect_uri={self.redirect_uri}&"
            f"response_type=code&"
            f"scope={'%20'.join(scopes)}&"
            f"force_verify=true"
        )
        
        print(f"\n📋 请在浏览器中授权以下权限: {'、'.join(scopes)}")
        print(f"🔗 授权URL: {auth_url}")
        print(f"🔄 重定向URI: {self.redirect_uri}")
        
        # 启动本地HTTP服务器等待回调
        with socketserver.TCPServer(('', self.server_port), OAuthHandler) as httpd:
            print(f"✅ 本地服务器已启动在端口 {self.server_port}")
            print("⏳ 等待Twitch回调...")
            
            # 自动打开浏览器
            try:
                webbrowser.open(auth_url)
                print("🌐 已尝试自动打开浏览器进行授权")
            except Exception as e:
                print(f"⚠️ 警告: 无法自动打开浏览器，请手动复制上面的URL到浏览器中。错误: {e}")
            
            # 等待授权码，超时时间设为5分钟
            start_time = time.time()
            timeout = 300  # 5分钟
            
            while auth_code is None:
                httpd.handle_request()
                if time.time() - start_time > timeout:
                    print("❌ 错误: 授权超时，请重新运行脚本并在5分钟内完成授权")
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
            
            print("🔄 正在交换访问令牌...")
            try:
                response = requests.post(token_url, params=token_params)
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data.get('access_token')
                    self.refresh_token = token_data.get('refresh_token')
                    print("✅ 成功获取访问令牌和刷新令牌！")
                    print(f"⏱️  令牌有效期: {token_data.get('expires_in')}秒")
                    print(f"📋 获得的权限范围: {token_data.get('scope')}")
                    return True, token_data
                else:
                    print(f"❌ 错误: 获取访问令牌失败，状态码: {response.status_code}")
                    print(f"📝 响应内容: {response.text}")
                    print("🔧 可能的原因：")
                    print("1. 重定向URI与Twitch开发者控制台中配置的不匹配")
                    print("2. 授权码已过期或无效")
                    print("3. 客户端ID或客户端密钥错误")
                    print("4. 网络连接问题")
                    return False, None
            except Exception as e:
                print(f"❌ 错误: 交换令牌过程中出现异常: {e}")
                return False, None
    
    def validate_token(self):
        """验证访问令牌的有效性"""
        if not self.access_token:
            print("❌ 错误: 没有可用的访问令牌进行验证")
            return False, None
            
        validate_url = 'https://id.twitch.tv/oauth2/validate'
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        try:
            response = requests.get(validate_url, headers=headers)
            
            if response.status_code == 200:
                print("✅ 访问令牌有效！")
                validate_data = response.json()
                print(f"👤 用户ID: {validate_data.get('user_id')}")
                print(f"📛 用户名: {validate_data.get('login')}")
                print(f"🔑 客户端ID: {validate_data.get('client_id')}")
                return True, validate_data
            else:
                print(f"❌ 错误: 访问令牌无效，状态码: {response.status_code}")
                print(f"📝 响应内容: {response.text}")
                return False, None
        except Exception as e:
            print(f"❌ 错误: 验证令牌过程中出现异常: {e}")
            return False, None
    
    def write_tokens_to_env(self, token_data):
        """将令牌写入.env文件"""
        try:
            # 读取现有.env文件
            if os.path.exists('.env'):
                with open('.env', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                lines = []
                
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
                
            print("✅ 已将令牌写入.env文件")
            return True
        except Exception as e:
            print(f"❌ 错误: 无法写入.env文件: {e}")
            # 提供备选保存方式
            print("🔧 备选方案: 手动将以下令牌添加到.env文件中：")
            for key, value in token_lines.items():
                print(f"{key}={value}")
            return False

def print_configuration_guide():
    """打印配置指南"""
    print("\n===== Twitch OAuth配置指南 =====")
    print("\n📋 当前配置信息：")
    print(f"🔑 客户端ID: {os.getenv('TWITCH_ID')}")
    print(f"🔄 重定向URI: http://localhost:3000/callback")
    print(f"🌐 HTTP服务器端口: 3000")
    
    print("\n🔧 故障排除步骤：")
    print("1. 确保Twitch开发者控制台中已添加正确的重定向URI")
    print("   - 访问: https://dev.twitch.tv/console/apps")
    print("   - 找到您的应用程序并点击'编辑'")
    print("   - 在'OAuth重定向URLs'部分添加: http://localhost:3000/callback")
    print("   - 点击'保存更改'")
    print("2. 确保.env文件中设置了正确的TWITCH_ID和TWITCH_SECRET")
    print("3. 确保端口3000未被其他程序占用")
    
    print("\n💡 重要提示：")
    print("- 重定向URI必须与Twitch开发者控制台中的配置完全匹配")
    print("- 包括协议(http/https)、主机名、端口和路径的每一个字符")
    print("- 即使是一个额外的斜杠或大小写不匹配也会导致redirect_mismatch错误")
    
    print("\n📌 如果需要使用其他端口：")
    print("- 修改代码中的DEFAULT_PORT变量")
    print("- 确保重定向URI中的端口号也相应更新")
    print("- 在Twitch开发者控制台中添加新的重定向URI")

def main():
    """主函数"""
    print("===== Twitch OAuth认证工具 (最终版) ======")
    
    # 确保requests库已安装
    try:
        import requests
    except ImportError:
        print("❌ 错误: requests库未安装")
        print("请运行 'pip install requests' 安装依赖")
        exit(1)
    
    # 默认配置
    DEFAULT_PORT = 3000
    DEFAULT_REDIRECT_URI = f'http://localhost:{DEFAULT_PORT}/callback'
    
    # 显示配置指南
    print_configuration_guide()
    
    # 询问是否需要修改端口
    change_port = input("\n是否需要修改端口号? (y/n): ")
    if change_port.lower() == 'y':
        try:
            new_port = int(input(f"请输入新的端口号 (当前: {DEFAULT_PORT}): "))
            DEFAULT_PORT = new_port
            DEFAULT_REDIRECT_URI = f'http://localhost:{DEFAULT_PORT}/callback'
            print(f"✅ 已更新配置：")
            print(f"   端口: {DEFAULT_PORT}")
            print(f"   重定向URI: {DEFAULT_REDIRECT_URI}")
        except ValueError:
            print("⚠️ 无效的端口号，使用默认端口")
    
    # 创建认证实例并开始认证流程
    twitch_auth = TwitchAuth(DEFAULT_REDIRECT_URI, DEFAULT_PORT)
    
    print("\n🚀 开始获取用户授权...")
    success, token_data = twitch_auth.obtain_user_tokens()
    
    if success and token_data:
        # 验证获取的令牌
        print("\n🔍 验证获取的令牌...")
        validate_success, _ = twitch_auth.validate_token()
        
        if validate_success:
            # 写入.env文件
            twitch_auth.write_tokens_to_env(token_data)
            print("\n🎉 认证流程已完成！")
            print("✅ 您的Twitch账号已成功授权该应用程序")
            print("📝 令牌已保存到.env文件中")
        else:
            print("\n⚠️ 警告: 令牌验证失败，但可能是临时问题")
            print("已尝试将令牌写入.env文件，请稍后手动验证")
            twitch_auth.write_tokens_to_env(token_data)
    else:
        print("\n❌ 认证流程未完成！")
        print("🔧 请按照以下步骤解决问题：")
        print("1. 确保Twitch开发者控制台中注册了正确的重定向URI")
        print("2. 检查.env文件中的TWITCH_ID和TWITCH_SECRET是否正确")
        print("3. 确保网络连接正常")
        print("4. 尝试使用不同的浏览器进行授权")
        print("5. 如果问题持续，考虑创建一个新的Twitch应用程序")
    
    # 最终提示
    print("\n===== 提示 =====")
    print("如果您需要添加更多的权限scope，请修改代码中的scopes列表")
    print("但请记住，添加更多scope可能会增加授权失败的风险")
    print("建议先使用基础scope完成授权，确认成功后再逐步添加其他权限")

if __name__ == '__main__':
    main()