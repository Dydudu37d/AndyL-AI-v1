import os

def remove_markdown_tokens_improved(log_file_path):
    """更彻底地移除日志文件中的所有</think>标记"""
    # 创建临时文件路径
    temp_file_path = log_file_path + '.tmp'
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file, open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            for line in file:
                # 完全移除所有包含</think>的行
                if '</think>' in line:
                    # 如果行中只有</think>，就跳过
                    if line.strip() == '</think>':
                        continue
                    # 否则，移除行中的所有</think>标记
                    cleaned_line = line.replace('</think>', '')
                    # 如果清理后的行不为空，才写入
                    if cleaned_line.strip():
                        temp_file.write(cleaned_line)
                else:
                    # 不包含标记的行直接写入
                    temp_file.write(line)
        
        # 用临时文件替换原文件
        os.replace(temp_file_path, log_file_path)
        print(f"已成功移除 {log_file_path} 中的所有</think>标记")
        return True
    except Exception as e:
        print(f"处理文件时出错: {e}")
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False

if __name__ == "__main__":
    log_file = "g:\\AndyL AI v1\\ai_computer_control.log"
    remove_markdown_tokens_improved(log_file)