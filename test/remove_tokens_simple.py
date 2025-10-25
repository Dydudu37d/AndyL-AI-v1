import os

def remove_tokens_simple(file_path):
    """简单直接地移除文件中的所有</think>标记"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除所有</think>标记
        cleaned_content = content.replace('</think>', '')
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"已成功移除 {file_path} 中的所有</think>标记")
        return True
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

if __name__ == "__main__":
    log_file = "g:\\AndyL AI v1\\ai_computer_control.log"
    remove_tokens_simple(log_file)