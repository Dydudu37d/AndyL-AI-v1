# 创建一个完全清理的日志文件
source_file = 'ai_computer_control.log'
target_file = 'ai_computer_control_clean.log'

with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'w', encoding='utf-8') as tgt:
    line_count = 0
    removed_count = 0
    
    for line in src:
        line_count += 1
        
        # 检查是否包含</think>标记
        if '</think>' in line:
            # 如果整行只有这个标记，完全跳过
            if line.strip() == '</think>':
                removed_count += 1
                continue
            # 否则，移除标记后写入
            cleaned_line = line.replace('</think>', '')
            tgt.write(cleaned_line)
        else:
            # 不包含标记的行直接写入
            tgt.write(line)
    
    print(f"处理完成！总行数: {line_count}, 移除的标记数: {removed_count}")
    print(f"清理后的日志已保存至: {target_file}")