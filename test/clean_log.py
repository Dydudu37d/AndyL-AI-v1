# 简单的日志清理脚本
with open('ai_computer_control.log', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned_content = content.replace('</think>', '')

with open('ai_computer_control.log', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print('已成功清理所有</think>标记')