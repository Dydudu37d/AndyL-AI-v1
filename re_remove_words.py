import re

"""
使用re库移除特定词语的示例程序

这个文件展示了如何使用Python的re库来移除文本中的特定词语，
包括基本用法和一些常见场景下的应用。
"""

# 定义一个函数，用于打印原始文本和处理后的文本
def show_result(original, processed, method_name):
    print(f"\n=== {method_name} ===")
    print(f"原始文本: {original}")
    print(f"处理后文本: {processed}")

# 示例1: 移除单个特定词语
def remove_single_word(text, word_to_remove):
    # 使用re.sub将特定词语替换为空字符串
    # 对于中文文本，不使用单词边界，直接匹配字符串
    pattern = re.escape(word_to_remove)
    result = re.sub(pattern, '', text)
    # 移除多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result

# 示例2: 移除多个特定词语
def remove_multiple_words(text, words_to_remove):
    # 构建正则表达式模式，使用|连接多个词语
    # re.escape确保特殊字符被正确处理
    patterns = [re.escape(word) for word in words_to_remove]
    pattern = '|'.join(patterns)
    result = re.sub(pattern, '', text)
    # 移除多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result

# 示例3: 移除词语（不区分大小写）
def remove_words_case_insensitive(text, words_to_remove):
    # 使用flags=re.IGNORECASE参数来忽略大小写
    patterns = [re.escape(word) for word in words_to_remove]
    pattern = '|'.join(patterns)
    result = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # 移除多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result

# 示例4: 移除词语及其变体（如复数形式）
def remove_words_with_variants(text, base_words):
    # 这里使用简单的方式处理常见变体
    # 实际应用中可能需要更复杂的规则
    patterns = []
    for word in base_words:
        # 对于英文处理可能的复数形式（加s或es）
        patterns.append(re.escape(word) + r'(s|es)?')
    pattern = '|'.join(patterns)
    result = re.sub(pattern, '', text)
    # 移除多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result

# 示例5: 实际应用 - 移除JSON格式中的特定标记
# 这个示例与tts_speaker.py中的场景相关
def remove_json_tokens(text, tokens_to_remove=['true', 'false', 'null', '*\'\'*', "''", ', true,']):
    # 移除特定的JSON标记
    patterns = [re.escape(token) for token in tokens_to_remove]
    pattern = '|'.join(patterns)
    result = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 移除额外的JSON相关标记，如OK, true, 0格式的内容
    # 匹配形如 "OK, true, 0" 或 "[OK, true, 0]" 的模式
    result = re.sub(r'\[?\s*OK\s*,\s*true\s*,\s*\d+\s*\]?', '', result, flags=re.IGNORECASE)
    
    # 移除单个数字（如果它们是孤立的标记）
    result = re.sub(r'\b\d+\b', '', result)
    
    # 移除多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    
    # 移除多余的逗号
    if result and result[-1] == ",":
        result = result[:-1]
    
    # 移除文本开头或结尾的逗号
    result = re.sub(r'^,\s*|\s*,\s*$', '', result)
    
    return result

# 示例6: 实际应用 - 从完整响应中提取纯文本
# 假设响应格式为：[文本内容, [表情名称,是否激活,过渡时间]]
def extract_plain_text_from_response(text):
    # 尝试提取第一个[和第二个[之间的文本内容
    plain_text = text  # 默认返回原始文本
    try:
        if '[' in text and text.count('[') >= 2:
            first_bracket = text.index('[')
            second_bracket = text.find('[', first_bracket + 1)
            if second_bracket != -1:
                # 提取文本内容并去除可能的引号和空格
                plain_text = text[first_bracket + 1:second_bracket - 1].strip()
                # 移除可能的引号
                plain_text = re.sub(r'^["\']|["\']$', '', plain_text)
                # 移除末尾的逗号
                if plain_text and plain_text[-1] == ",":
                    plain_text = plain_text[:-1]
        
        # 额外处理：移除末尾的不需要内容，如', 0.3'
        # 匹配形如 ", 数字.数字" 或 ", 数字" 的模式
        plain_text = re.sub(r',\s*\d+(\.\d+)?$', '', plain_text)
        # 再次移除末尾可能的逗号
        if plain_text and plain_text[-1] == ",":
            plain_text = plain_text[:-1]
    except Exception as e:
        print(f"解析错误: {e}")
    # 返回处理后的文本
    return plain_text.strip()

# 主函数，用于测试所有示例
def main():
    # 测试文本
    test_text = "你好，我是AndyL，这是一个测试文本。Hello, this is a test text with some words to remove."
    json_response_text = "[你好，我是AndyL，这是测试内容, ['Happy', true, 0.3]]"
    
    # 测试示例1
    result1 = remove_single_word(test_text, "测试")
    show_result(test_text, result1, "移除单个词语")
    
    # 测试示例2
    words_to_remove = ["测试", "test", "AndyL"]
    result2 = remove_multiple_words(test_text, words_to_remove)
    show_result(test_text, result2, "移除多个词语")
    
    # 测试示例3
    words_to_remove = ["hello", "andyl"]
    result3 = remove_words_case_insensitive(test_text, words_to_remove)
    show_result(test_text, result3, "移除词语（不区分大小写）")
    
    # 测试示例4
    base_words = ["测试", "text"]
    result4 = remove_words_with_variants(test_text, base_words)
    show_result(test_text, result4, "移除词语及其变体")
    
    # 测试示例5
    result5 = remove_json_tokens(json_response_text)
    show_result(json_response_text, result5, "移除JSON标记")
    
    # 测试示例6
    result6 = extract_plain_text_from_response(json_response_text)
    show_result(json_response_text, result6, "从响应中提取纯文本")

if __name__ == "__main__":
    main()