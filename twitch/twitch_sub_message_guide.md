# Twitch订阅文字消息读取指南

本文档详细介绍如何使用`twitch_sub_message_reader.py`脚本读取Twitch订阅的文字消息，并提供相关的技术说明和自定义扩展方法。

## 工具概述

`twitch_sub_message_reader.py`是一个专门设计用于接收和显示Twitch订阅通知及文字消息的工具。该工具提供两种接收方式：

1. **IRC聊天方式** - 简单直接，通过连接到Twitch IRC服务器监听订阅事件
2. **EventSub Webhook方式** - 更强大和灵活，通过设置Webhook接收Twitch推送的订阅事件

两种方式都能够读取订阅者发送的文字消息，并将其显示在控制台和日志文件中。

## 安装和配置

### 前提条件

- Python 3.6或更高版本
- 已安装必要的依赖库：`requests`, `python-dotenv`

### 安装依赖

```bash
pip install requests python-dotenv
```

### 环境变量配置

在项目根目录创建`.env`文件，并添加以下环境变量：

```
# 所有方式都需要的配置
TWITCH_ID=your_twitch_client_id
TWITCH_OAUTH_TOKEN=your_twitch_oauth_token

# EventSub方式额外需要的配置
TWITCH_SECRET=your_twitch_client_secret
```

**获取方法：**
- `TWITCH_ID`和`TWITCH_SECRET`：通过[Twitch开发者控制台](https://dev.twitch.tv/console/apps)创建应用获取
- `TWITCH_OAUTH_TOKEN`：使用[Twitch授权页面](https://id.twitch.tv/oauth2/authorize)或第三方工具（如[Twitch Chat OAuth Password Generator](https://twitchapps.com/tmi/)）获取

## 使用方法

### IRC聊天方式

这是最简单的方式，适用于只需要基本订阅通知的用户。

```bash
python twitch_sub_message_reader.py irc
```

运行后，工具会连接到您的Twitch聊天室，并在有新订阅时显示订阅信息和文字消息。

### EventSub Webhook方式

这种方式功能更强大，但配置稍复杂，需要一个公开可访问的HTTPS URL。

```bash
python twitch_sub_message_reader.py eventsub <callback_url> [port]
```

- `<callback_url>`：必需，公开可访问的HTTPS URL，用于接收Twitch的事件通知
- `[port]`：可选，本地服务器监听的端口，默认8080

**本地测试提示：**
- 对于本地测试，可以使用[ngrok](https://ngrok.com/)创建临时的公开URL
- 示例：`python twitch_sub_message_reader.py eventsub https://your-ngrok-url.ngrok.io/callback 8080`

## 订阅消息的结构

### IRC方式的订阅消息结构

通过IRC接收的订阅消息是`USERNOTICE`类型的消息，包含以下关键信息：

- `display-name`：订阅者的显示名称
- `msg-param-sub-plan`：订阅等级（1000、2000、3000或Prime）
- 消息内容：订阅者输入的文字消息

在脚本中，这些信息通过以下代码解析：

```python
# 解析订阅者名称
for part in parts:
    if part.startswith('display-name='):
        subscriber_name = part.split('=')[1]
        break

# 获取订阅等级
for part in parts:
    if part.startswith('msg-param-sub-plan='):
        sub_tier = part.split('=')[1]
        break

# 获取订阅消息
if ':' in data:
    message_parts = data.split(':', 2)
    if len(message_parts) > 2:
        message = message_parts[2].strip()
```

### EventSub方式的订阅消息结构

通过EventSub接收的订阅事件是JSON格式的数据，包含以下关键信息：

```json
{
  "subscription": {
    "type": "channel.subscribe",
    "version": "1",
    "condition": {"broadcaster_user_id": "12345"}
  },
  "event": {
    "user_id": "67890",
    "user_name": "subscriber_name",
    "broadcaster_user_id": "12345",
    "tier": "1000",
    "message": "这是订阅者的文字消息"
  }
}
```

在脚本中，这些信息通过以下代码解析：

```python
# 解析事件数据
event_data = json.loads(post_data.decode('utf-8'))
event = event_data.get('event', {})
subscriber_name = event.get('user_name')
subscriber_id = event.get('user_id')
tier = event.get('tier')
message = event.get('message')
```

## 自定义订阅消息处理

您可以根据自己的需求修改脚本，自定义订阅消息的处理逻辑。以下是一些常见的自定义方式：

### 1. 修改消息显示格式

在`_parse_and_display_irc_sub_message`和`EventSubHandler.do_POST`方法中，可以自定义订阅消息的显示格式。

### 2. 添加自动回复功能

对于IRC方式，可以添加自动发送感谢消息的功能：

```python
# 在_parse_and_display_irc_sub_message方法末尾添加
# 注意：需要额外的权限和代码来发送消息
def send_thank_you(irc, channel, subscriber_name):
    thank_you_message = f"PRIVMSG #{channel} :感谢@{subscriber_name}的订阅！\r\n"
    irc.send(thank_you_message.encode('utf-8'))

# 使用示例
send_thank_you(irc, self.user_name, subscriber_name)
```

### 3. 集成到其他系统

您可以将订阅信息发送到其他系统或服务，例如数据库、通知服务等：

```python
# 示例：将订阅信息保存到数据库
def save_to_database(subscriber_name, tier, message):
    import sqlite3
    conn = sqlite3.connect('subscriptions.db')
    c = conn.cursor()
    c.execute("INSERT INTO subscriptions (name, tier, message, timestamp) VALUES (?, ?, ?, datetime('now'))",
              (subscriber_name, tier, message))
    conn.commit()
    conn.close()

# 使用示例
save_to_database(subscriber_name, sub_tier, message)
```

### 4. 添加声音提醒

可以在收到订阅时播放声音提醒：

```python
# 示例：播放声音提醒
def play_sound():
    import winsound  # Windows系统
    winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms

# 使用示例
play_sound()
```

## 两种接收方式的比较

| 特性 | IRC聊天方式 | EventSub Webhook方式 |
|------|------------|---------------------|
| 配置复杂度 | 低 | 中高 |
| 所需资源 | 低 | 中 |
| 实时性 | 高 | 高 |
| 可靠性 | 一般 | 高 |
| 支持的事件类型 | 基本（订阅、礼物订阅） | 丰富（订阅、礼物、捐赠、关注等） |
| 公开URL要求 | 不需要 | 需要公开可访问的HTTPS URL |
| 数据丰富度 | 基本信息 | 详细信息 |

## 常见问题及解决方案

### 1. 无法连接到IRC服务器

**可能原因：**
- OAuth令牌无效或过期
- 防火墙阻止了连接

**解决方案：**
- 重新生成OAuth令牌
- 检查防火墙设置，确保允许连接到`irc.chat.twitch.tv:6667`

### 2. EventSub验证失败

**可能原因：**
- 回调URL不是公开可访问的
- 回调URL不是HTTPS
- 端口不是Twitch支持的端口（80、443、8080或8443）

**解决方案：**
- 使用ngrok等工具创建临时的公开HTTPS URL
- 确保使用Twitch支持的端口

### 3. 看不到订阅消息

**可能原因：**
- 订阅者没有输入文字消息
- 权限不足，无法接收订阅通知

**解决方案：**
- 确保OAuth令牌具有`channel:read:subscriptions`权限
- 测试时让订阅者输入订阅消息

### 4. 程序运行一段时间后断开连接

**可能原因：**
- 网络不稳定
- Twitch IRC服务器超时

**解决方案：**
- 添加自动重连机制
- 确保网络连接稳定

## 扩展功能建议

1. **多频道支持**：修改脚本支持同时监控多个频道的订阅消息
2. **历史记录查询**：添加查询历史订阅记录的功能
3. **统计分析**：添加订阅数据统计和分析功能
4. **Web界面**：开发简单的Web界面展示实时订阅信息
5. **多语言支持**：添加多语言界面支持

## 完整流程图

下面是使用该工具读取订阅文字消息的完整流程图：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  启动工具并选择 │     │   连接到Twitch  │     │  接收订阅事件  │
│   接收方式      │────▶│   IRC服务器或   │────▶│  并解析消息    │
└─────────────────┘     │   设置EventSub  │     └────────┬────────┘
                        │   Webhook       │              │
                        └─────────────────┘              │
                                                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   显示订阅信息  │◀────│   处理订阅消息  │
                        │   和文字消息    │     │   （自定义逻辑）│
                        └─────────────────┘     └─────────────────┘
```

## 总结

`twitch_sub_message_reader.py`提供了两种有效的方式来读取Twitch订阅的文字消息。根据您的需求和技术能力，可以选择适合的方式来集成订阅消息功能到您的Twitch直播或相关应用中。

通过自定义处理逻辑，您可以实现各种实用功能，如自动感谢、数据统计、通知提醒等，提升您的直播体验和与观众的互动质量。