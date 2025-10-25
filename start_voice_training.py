#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AndyL AI 语音训练快速启动脚本

此脚本用于指导用户如何训练语音模型并让它生成语音。
它将引导用户完成整个训练流程，包括训练状态检查、
模型训练、pth格式导出和生成音频样本。
"""

import os
import sys
import time
import subprocess

# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义全局路径
BASE_DIR = r'g:\AndyL AI v1'
TRAINING_AUDIO_DIR = os.path.join(BASE_DIR, 'AndyL_say', 'training_audio')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'train_and_export_voice.py')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_title(title):
    """打印带颜色的标题"""
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}{title}{bcolors.ENDC}\n")


def print_info(info):
    """打印信息"""
    print(f"{bcolors.OKBLUE}ℹ️ {info}{bcolors.ENDC}")


def print_success(success):
    """打印成功信息"""
    print(f"{bcolors.OKGREEN}✅ {success}{bcolors.ENDC}")


def print_warning(warning):
    """打印警告信息"""
    print(f"{bcolors.WARNING}⚠️ {warning}{bcolors.ENDC}")


def print_error(error):
    """打印错误信息"""
    print(f"{bcolors.FAIL}❌ {error}{bcolors.ENDC}")


def print_step(step_num, step_title):
    """打印步骤信息"""
    print(f"\n{bcolors.BOLD}{step_num}. {step_title}{bcolors.ENDC}")


def count_training_files():
    """计算训练音频文件数量"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    count = 0
    
    try:
        if os.path.exists(TRAINING_AUDIO_DIR):
            for file in os.listdir(TRAINING_AUDIO_DIR):
                _, ext = os.path.splitext(file)
                if ext.lower() in audio_extensions:
                    count += 1
    except Exception as e:
        print_error(f"无法读取训练音频目录: {str(e)}")
    
    return count


def check_environment():
    """检查环境是否准备就绪"""
    print_title("检查训练环境")
    
    # 检查训练音频文件
    file_count = count_training_files()
    print_info(f"在 {TRAINING_AUDIO_DIR} 中找到了 {file_count} 个音频文件")
    
    if file_count < 10:
        print_warning("警告: 训练音频文件数量较少，可能影响训练效果。建议至少准备20个不同的音频样本。")
    elif file_count < 50:
        print_info("音频文件数量适中，可以进行基础训练。")
    else:
        print_success("音频文件数量充足，有利于获得更好的训练效果。")
    
    # 检查训练脚本是否存在
    if not os.path.exists(TRAIN_SCRIPT):
        print_error(f"未找到训练脚本: {TRAIN_SCRIPT}")
        print_info("请确保train_and_export_voice.py文件存在于工作目录中")
        return False
    
    print_success("环境检查完成，一切就绪！")
    return True


def train_voice_model():
    """执行语音模型训练"""
    print_title("开始语音模型训练")
    
    try:
        print_info("即将启动语音训练流程...")
        print_info("这个过程将包括:")
        print_info("1. 训练数据验证")
        print_info("2. 语音模型训练")
        print_info("3. pth格式模型导出")
        print_info("4. 音频样本生成")
        print_info("5. 训练报告生成")
        
        print_warning("注意: 如果模型已经训练过，系统会询问是否重新训练。")
        print_warning("      如需重新训练，请在提示时输入 'y'。")
        
        input("\n按Enter键开始训练...")
        
        # 运行训练脚本
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            cwd=BASE_DIR,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 打印训练脚本的输出
        print(f"\n{result.stdout}")
        
        if result.returncode != 0:
            print_error(f"训练过程中出现错误: {result.stderr}")
            return False
        
        print_success("语音模型训练流程已完成！")
        return True
    except Exception as e:
        print_error(f"执行训练脚本时发生错误: {str(e)}")
        return False


def verify_training_results():
    """验证训练结果"""
    print_title("验证训练结果")
    
    # 检查生成的文件
    models_dir = os.path.join(BASE_DIR, 'models')
    exported_dir = os.path.join(BASE_DIR, 'exported_voice_models')
    test_audio_dir = os.path.join(BASE_DIR, 'AndyL_say', 'test_audio')
    reports_dir = os.path.join(BASE_DIR, 'reports')
    
    # 检查模型文件
    best_model = os.path.join(models_dir, 'best_model.pth')
    if os.path.exists(best_model):
        size = os.path.getsize(best_model) / 1024  # KB
        print_success(f"找到训练好的模型文件: {best_model} ({size:.2f} KB)")
    else:
        print_warning(f"未找到模型文件: {best_model}")
    
    # 检查导出的pth模型
    latest_pth = os.path.join(exported_dir, 'andyL_voice_model_latest.pth')
    if os.path.exists(latest_pth):
        size = os.path.getsize(latest_pth) / 1024  # KB
        print_success(f"找到导出的pth模型: {latest_pth} ({size:.2f} KB)")
    else:
        print_warning(f"未找到导出的pth模型: {latest_pth}")
    
    # 检查音频样本
    if os.path.exists(test_audio_dir):
        audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith(('.wav', '.mp3'))]
        if audio_files:
            print_success(f"找到 {len(audio_files)} 个生成的音频样本，保存在: {test_audio_dir}")
        else:
            print_warning(f"未找到音频样本文件")
    
    # 检查训练报告
    if os.path.exists(reports_dir):
        report_files = [f for f in os.listdir(reports_dir) if f.startswith('voice_training_report')]
        if report_files:
            # 按修改时间排序，获取最新的报告
            report_files.sort(key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)), reverse=True)
            latest_report = os.path.join(reports_dir, report_files[0])
            print_success(f"找到最新的训练报告: {latest_report}")
        else:
            print_warning(f"未找到训练报告")


def show_next_steps():
    """显示后续操作建议"""
    print_title("后续操作建议")
    
    print_info("1. 查看训练报告了解详细的训练情况")
    print_info("2. 播放生成的音频样本，检查语音质量")
    print_info("3. 将导出的pth模型用于语音合成")
    print_info("4. 如果对效果不满意，可以添加更多音频样本并重新训练")
    
    print_warning("\n提示:")
    print_warning("- 要再次训练模型，只需重新运行此脚本")
    print_warning("- 要单独测试音频播放，可以运行 test_audio_playback.py 脚本")
    print_warning("- 详细的使用说明请参考 '语音训练完整指南.md'")


def main():
    """主函数"""
    print(f"{bcolors.HEADER}=" * 70)
    print(f"{bcolors.HEADER}{bcolors.BOLD}      AndyL AI 语音训练快速启动工具      {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=" * 70)
    print(f"{bcolors.OKBLUE}此工具将帮助您训练语音模型并让它生成语音。{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}训练音频文件已检测到位于: {TRAINING_AUDIO_DIR}{bcolors.ENDC}")
    print(f"{bcolors.HEADER}=" * 70)
    
    # 步骤1: 检查环境
    if not check_environment():
        print_error("环境检查未通过，无法继续训练")
        sys.exit(1)
    
    # 步骤2: 训练语音模型
    if not train_voice_model():
        print_error("语音模型训练失败")
        sys.exit(1)
    
    # 步骤3: 验证训练结果
    verify_training_results()
    
    # 步骤4: 显示后续操作建议
    show_next_steps()
    
    print(f"\n{bcolors.OKGREEN}🎉 语音训练流程已全部完成！{bcolors.ENDC}")
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()