import json

# 尝试不同的编码方式读取日志文件
def read_log_file_with_encodings(file_path):
    encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read file with encoding: {encoding}")
            return content, encoding
        except Exception as e:
            print(f"Error reading with {encoding}: {e}")
    return "", ""

# 读取vtube_studio_controller.log文件
log_content, encoding = read_log_file_with_encodings('vtube_studio_controller.log')
print(f"\nVTubeStudioController Log (using {encoding}):\n")

if log_content:
    # 显示最后3000个字符
    print(log_content[-3000:])
    
    # 查找热键相关的信息
    print("\n\nSearching for hotkey information in logs...\n")
    if 'hotkeys' in log_content:
        start_index = log_content.rfind('"hotkeys"')
        if start_index > 0:
            # 尝试找到热键数组的开始和结束位置
            end_index = log_content.find(']}', start_index)
            if end_index > start_index:
                print("Found hotkey information:")
                print(log_content[start_index:end_index+2])
    elif 'Hotkey' in log_content:
        print("Found mentions of Hotkey, but not the full list. Here are relevant parts:")
        # 提取包含Hotkey的行
        lines = log_content.split('\n')
        for line in lines:
            if 'Hotkey' in line:
                print(line)
else:
    print("Could not read log file with any supported encoding.")