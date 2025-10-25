import requests
import os
import dotenv

def load_env():
    """加载环境变量"""
    dotenv.load_dotenv()
    client_id = os.getenv('TWITCH_ID')
    client_secret = os.getenv('TWITCH_SECRET')
    
    if not client_id or not client_secret:
        print("错误：请确保在.env文件中设置了TWITCH_ID和TWITCH_SECRET")
        exit(1)
    
    return client_id, client_secret

def test_app_access_token(client_id, client_secret):
    """测试应用访问令牌（验证client_id和client_secret是否有效）"""
    print("\n=== 测试应用访问令牌 ===")
    token_url = "https://id.twitch.tv/oauth2/token"
    token_params = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
        'scope': ''
    }
    
    try:
        response = requests.post(token_url, params=token_params)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✓ 成功获取应用访问令牌！")
            print("这表明您的client_id和client_secret是有效的")
            return True
        else:
            print(f"✗ 无法获取应用访问令牌")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"获取令牌时出错: {e}")
        return False

def generate_auth_url(client_id, scopes, port=3009):
    """生成授权URL"""
    redirect_uri = f'http://localhost:{port}/callback'
    auth_url = (
        f"https://id.twitch.tv/oauth2/authorize?" \
        f"client_id={client_id}&" \
        f"redirect_uri={redirect_uri}&" \
        f"response_type=code&" \
        f"scope={' '.join(scopes)}&" \
        f"force_verify=true"
    )
    return auth_url, redirect_uri

def test_scope_individually(client_id):
    """逐个测试scope，帮助找出无效的scope"""
    print("\n=== 逐个测试Scope ===")
    print("这个功能将帮助您找出可能导致问题的scope")
    
    # 常见的Twitch scope列表
    common_scopes = [
        'user:read:email',
        'chat:read',
        'chat:edit',
        'channel:read:redemptions',
        'channel:moderate',
        'whispers:read',
        'whispers:send',  # 这是错误消息中提到的scope
        'channel:read:stream_key',
        'channel:manage:broadcast',
        'channel:manage:schedule',
        'user:read:chat',
        'user:write:chat'
    ]
    
    print("\n测试以下scope：")
    for i, scope in enumerate(common_scopes, 1):
        print(f"{i}. {scope}")
    
    # 为每个scope生成一个单独的测试URL
    print("\n=== 生成的测试URL ===")
    for i, scope in enumerate(common_scopes, 1):
        auth_url, _ = generate_auth_url(client_id, [scope])
        print(f"\n测试URL {i} (仅包含scope: {scope}):")
        print(auth_url)
    
    print("\n使用说明:")
    print("1. 复制上面的URL，一个一个在浏览器中打开")
    print("2. 如果某个URL在浏览器中显示'无效scope'错误，那么这个scope就是问题所在")
    print("3. 特别是关注第6个URL (whispers:send)，这是错误消息中提到的scope")
    print("4. 一旦找到问题scope，请从您的主程序中移除它")

def test_simplified_scope_combinations(client_id):
    """测试简化的scope组合"""
    print("\n=== 测试简化的Scope组合 ===")
    
    # 定义几个简化的scope组合
    scope_combinations = [
        {
            'name': '最基础组合',
            'scopes': ['user:read:email', 'chat:read', 'chat:edit']
        },
        {
            'name': '扩展组合A',
            'scopes': ['user:read:email', 'chat:read', 'chat:edit', 'channel:read:redemptions', 'channel:moderate']
        },
        {
            'name': '排除whispers:send的完整组合',
            'scopes': [
                'user:read:email', 'chat:read', 'chat:edit', 
                'channel:read:redemptions', 'channel:moderate', 
                'whispers:read', 'channel:read:stream_key', 
                'channel:manage:broadcast', 'channel:manage:schedule', 
                'user:read:chat', 'user:write:chat'
            ]
        }
    ]
    
    # 为每个组合生成测试URL
    print("\n=== 生成的测试URL ===")
    for i, combo in enumerate(scope_combinations, 1):
        auth_url, _ = generate_auth_url(client_id, combo['scopes'])
        print(f"\n测试URL {i} ({combo['name']}):")
        print(f"包含scope: {', '.join(combo['scopes'])}")
        print(auth_url)
    
    print("\n使用建议:")
    print("1. 先测试第1个URL (最基础组合)，如果成功，再测试更复杂的组合")
    print("2. 第3个组合排除了'whispers:send'，这可能解决您遇到的错误")
    print("3. 一旦找到一个可以工作的组合，您可以在主程序中使用这个组合")

def print_troubleshooting_guide():
    """打印故障排除指南"""
    print("\n===== Twitch OAuth Scope故障排除指南 =====")
    print("\n常见问题和解决方案:")
    print("\n1. 'invalid scope requested: 'whispers:send' 错误")
    print("   - 原因: 'whispers:send' scope可能需要特殊权限或已被Twitch更改")
    print("   - 解决方案: 从您的scope列表中移除'whispers:send'")
    
    print("\n2. 端口占用问题")
    print("   - 原因: 您尝试使用的端口已经被其他程序占用")
    print("   - 解决方案: 更换一个未被使用的端口，如3009、3010等")
    print("   - 确保重定向URI中的端口与程序使用的端口一致")
    
    print("\n3. 重定向URI不匹配错误")
    print("   - 原因: 程序中使用的重定向URI与Twitch开发者控制台中配置的不一致")
    print("   - 解决方案: 确保两者完全一致，包括端口号")
    
    print("\n4. 授权后没有接收到回调")
    print("   - 原因: 本地服务器可能没有正确启动或被防火墙阻止")
    print("   - 解决方案: 检查防火墙设置，确保端口已开放")
    
    print("\n5. scope相关的其他错误")
    print("   - 原因: Twitch定期更新其API和支持的scope")
    print("   - 解决方案: 参考最新的Twitch API文档，确保使用的scope仍然受支持")
    print("   - 文档地址: https://dev.twitch.tv/docs/authentication/scopes/")
    
    print("\n修复建议:")
    print("1. 如果您的程序只需要基本功能，使用最简化的scope组合")
    print("2. 对于'whispers:send'错误，最简单的解决方案是从scope列表中移除它")
    print("3. 如果需要发送私信功能，请查看Twitch最新的API文档，了解当前支持的相关scope")

def main():
    """主函数"""
    print("===== Twitch OAuth Scope诊断工具 =====")
    print("此工具帮助您诊断和解决Twitch OAuth认证中的'scope'相关问题")
    
    # 加载环境变量
    client_id, client_secret = load_env()
    
    # 测试应用访问令牌（验证client_id和client_secret）
    app_token_success = test_app_access_token(client_id, client_secret)
    
    if app_token_success:
        # 打印当前诊断的问题
        print("\n=== 诊断您报告的问题 ===")
        print("您遇到了错误: {\"status\":400,\"message\":\"invalid scope requested: 'whispers:send'\"}")
        print("这表明您请求了一个无效的scope: 'whispers:send'")
        
        # 提供解决方案选项
        print("\n=== 解决方案选项 ===")
        print("1. 从您的scope列表中移除'whispers:send'")
        print("2. 测试其他scope组合，找出哪些可以正常工作")
        
        # 生成测试URL
        test_simplified_scope_combinations(client_id)
        test_scope_individually(client_id)
        
        # 打印故障排除指南
        print_troubleshooting_guide()
        
        print("\n===== 诊断完成 =====")
        print("根据上面的测试结果，您可以修改主程序中的scope列表，移除无效的scope")
        print("建议先尝试排除'whispers:send'的scope组合")
    else:
        print("\n===== 诊断中断 =====")
        print("请先确保您的TWITCH_ID和TWITCH_SECRET设置正确")
        print("然后再次运行此工具")

if __name__ == '__main__':
    main()