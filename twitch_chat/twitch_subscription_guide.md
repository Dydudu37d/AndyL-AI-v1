# Twitch订阅通知接收指南

本指南将详细介绍如何使用`twitch_subscription_handler.py`脚本接收Twitch上的订阅通知。

## 脚本概述

`twitch_subscription_handler.py`提供了两种接收Twitch订阅通知的方法：

1. **IRC聊天方式** - 通过连接到Twitch聊天室直接接收订阅通知
2. **EventSub API方式** - 通过设置Webhook接收Twitch官方的订阅事件通知

## 前提条件

在使用本脚本之前，请确保您已完成以下准备工作：

1. 在Twitch开发者控制台创建应用，并获取`Client ID`和`Client Secret`
2. 在项目根目录创建`.env`文件，并设置以下环境变量：
   
   ```env
   # 必需的变量
   TWITCH_ID=your_twitch_client_id
   TWITCH_OAUTH_TOKEN=your_twitch_user_oauth_token
   
   # 使用EventSub方式时必需
   TWITCH_SECRET=your_twitch_client_secret
   
   # 可选变量
   TWITCH_REFRESH_TOKEN=your_twitch_refresh_token
   ```

3. 安装所需的Python依赖：
   
   ```bash
   pip install requests python-dotenv
   ```

## 方法一：IRC聊天方式

### 优点
- 设置简单，无需公开可访问的服务器
- 实时接收订阅通知
- 适合个人直播主使用

### 缺点
- 可能会遗漏某些订阅通知
- 信息格式有限
- 依赖IRC连接的稳定性

### 使用方法

```bash
python twitch_subscription_handler.py irc
```

运行后，脚本会连接到您的Twitch聊天室，并实时监听订阅通知。当有新的订阅时，会在控制台显示订阅信息。

## 方法二：EventSub API方式

### 优点
- 可靠的通知机制，由Twitch官方推送
- 包含更详细的订阅信息
- 支持更多类型的事件（可以扩展脚本支持其他事件）

### 缺点
- 设置较复杂，需要公开可访问的HTTPS URL
- 本地测试需要使用ngrok等工具
- 需要定期更新订阅（每3个月）

### 使用方法

1. **获取公开可访问的HTTPS URL**

   对于本地测试，您可以使用ngrok创建临时的公开URL：
   
   ```bash
   ngrok http 8080
   ```
   
   运行后，ngrok会提供一个类似`https://xxxx-xxx-xxx-xxx-xxx.ngrok-free.app`的URL。

2. **运行脚本**

   ```bash
   python twitch_subscription_handler.py eventsub https://your-public-url.com/callback 8080
   ```
   
   其中：
   - `https://your-public-url.com/callback` 是您的公开回调URL
   - `8080` 是本地服务器监听的端口

3. **在Twitch开发者控制台配置**

   确保您的Twitch应用已配置正确的重定向URL，并且您的OAuth令牌包含必要的权限。

## 自定义订阅处理逻辑

您可以通过修改`CustomSubscriptionHandler`类中的`_on_new_subscription`方法来自定义订阅通知的处理逻辑。例如：

- 发送感谢消息到聊天室
- 将订阅信息保存到数据库
- 播放特定的音效
- 触发其他自定义事件

## 常见问题排查

### IRC方式常见问题

1. **连接失败**
   - 检查OAuth令牌是否有效
   - 确保Twitch ID设置正确
   - 确认IRC端口（默认为6667）未被防火墙阻止

2. **收不到订阅通知**
   - 确保您的OAuth令牌包含`chat:read`权限
   - 检查您是否正确加入了自己的聊天室

### EventSub方式常见问题

1. **回调URL验证失败**
   - 确保回调URL是公开可访问的
   - 确认URL使用HTTPS协议
   - 检查端口是否为80、443、8080或8443之一

2. **收不到事件通知**
   - 检查服务器日志，确认是否收到了Twitch的请求
   - 验证EventSub订阅是否处于活动状态
   - 确保本地服务器正常运行

## 日志查看

脚本会生成`twitch_subscription.log`日志文件，记录所有的操作和错误信息。您可以通过查看此文件来排查问题。

## 扩展建议

1. 添加自动重连机制，确保长时间稳定运行
2. 实现订阅信息的持久化存储
3. 增加对其他Twitch事件的支持（如follows、cheers等）
4. 开发图形用户界面，方便非技术用户使用

## 注意事项

- 请妥善保管您的OAuth令牌和客户端密钥，不要分享给他人
- 定期更新您的令牌，以确保持续接收订阅通知
- 遵守Twitch的API使用政策，避免滥用

祝您使用愉快！如有任何问题，请查看脚本的日志文件或联系技术支持。