#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 控制系统执行器

该模块负责接收AI的控制指令并执行相应的鼠标和键盘操作。
支持的指令格式：
1. 管道分隔的字符串：'m_up|m_up|m_up|m_left|m_click|a|b|c'
2. 列表格式：['m_up', 'm_up', 'm_up', 'm_left', 'm_click', 'a', 'b', 'c']

指令类型：
- 鼠标移动：m_up, m_down, m_left, m_right
- 鼠标点击：m_click, m_right_click, m_double_click
- 鼠标滚轮：m_scroll_up, m_scroll_down
- 键盘输入：单个字符（如'a', 'B', '1'）或特殊键（如'enter', 'tab', 'space'）
- 组合键：以'+'连接的按键组合（如'ctrl+c', 'shift+a'）
"""

import os
import sys
import time
import logging
import json
from typing import List, Union, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_control.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AI_Control_Executor")

# 确保可以导入鼠标控制器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mouse_controller import get_mouse_controller


class AIControlExecutor:
    """AI控制指令执行器"""
    
    def __init__(self):
        """初始化执行器"""
        # 获取鼠标控制器实例
        self.mouse_controller = get_mouse_controller()
        self.mouse_controller.initialize()
        
        # 设置移动步长
        self.move_step = 10  # 每次移动的像素数
        
        # 设置特殊按键映射
        self.special_keys = {
            'enter': 'enter',
            'return': 'enter',
            'ctrl': 'ctrl',
            'control': 'ctrl',
            'alt': 'alt',
            'altgr': 'altgr',
            'shift': 'shift',
            'win': 'win',
            'windows': 'win',
            'tab': 'tab',
            'space': 'space',
            'spacebar': 'space',
            'escape': 'esc',
            'esc': 'esc',
            'delete': 'delete',
            'backspace': 'backspace',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'home': 'home',
            'end': 'end',
            'page_up': 'page_up',
            'page_down': 'page_down',
            'f1': 'f1',
            'f2': 'f2',
            'f3': 'f3',
            'f4': 'f4',
            'f5': 'f5',
            'f6': 'f6',
            'f7': 'f7',
            'f8': 'f8',
            'f9': 'f9',
            'f10': 'f10',
            'f11': 'f11',
            'f12': 'f12',
        }
        
        logger.info("AI控制执行器已初始化")
    
    def parse_commands(self, commands: Union[str, List[str]]) -> List[str]:
        """
        解析AI的指令为操作列表
        
        参数:
            commands: AI的指令，可以是管道分隔的字符串或列表
        
        返回:
            解析后的操作列表
        """
        if isinstance(commands, str):
            # 处理管道分隔的字符串
            return [cmd.strip() for cmd in commands.split('|') if cmd.strip()]
        elif isinstance(commands, list):
            # 处理列表格式
            return [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        else:
            logger.error(f"不支持的指令格式: {type(commands)}")
            return []
    
    def execute_command(self, command: str) -> bool:
        """
        执行单个指令
        
        参数:
            command: 单个控制指令
        
        返回:
            执行是否成功
        """
        command = command.lower()
        
        # 处理鼠标移动指令
        if command == 'm_up':
            x, y = self.mouse_controller.get_mouse_position()
            return self.mouse_controller.move_mouse(x, y - self.move_step, absolute=True)
        elif command == 'm_down':
            x, y = self.mouse_controller.get_mouse_position()
            return self.mouse_controller.move_mouse(x, y + self.move_step, absolute=True)
        elif command == 'm_left':
            x, y = self.mouse_controller.get_mouse_position()
            return self.mouse_controller.move_mouse(x - self.move_step, y, absolute=True)
        elif command == 'm_right':
            x, y = self.mouse_controller.get_mouse_position()
            return self.mouse_controller.move_mouse(x + self.move_step, y, absolute=True)
        
        # 处理鼠标点击指令
        elif command == 'm_click':
            return self.mouse_controller.click(button='left')
        elif command == 'm_right_click':
            return self.mouse_controller.click(button='right')
        elif command == 'm_double_click':
            return self.mouse_controller.click(button='left', count=2)
        
        # 处理鼠标滚轮指令
        elif command == 'm_scroll_up':
            return self.mouse_controller.scroll(y=1)
        elif command == 'm_scroll_down':
            return self.mouse_controller.scroll(y=-1)
        
        # 处理组合键指令
        elif '+' in command:
            keys = [k.strip() for k in command.split('+')]
            return self.mouse_controller.press_combination(keys)
        
        # 处理特殊按键
        elif command in self.special_keys:
            success = self.mouse_controller.press_key(self.special_keys[command])
            time.sleep(0.05)
            success = success and self.mouse_controller.release_key(self.special_keys[command])
            return success
        
        # 处理修饰键（ctrl, shift, alt等）
        elif hasattr(self.mouse_controller, 'key_mapping') and command in self.mouse_controller.key_mapping:
            success = self.mouse_controller.press_key(command)
            time.sleep(0.05)
            success = success and self.mouse_controller.release_key(command)
            return success
        
        # 处理文本输入命令
        elif command.startswith('t_'):
            # 提取t_后面的文本内容
            text = command[2:]
            return self.mouse_controller.type_string(text)
            
        # 处理普通键盘输入
        elif len(command) == 1:
            return self.mouse_controller.type_string(command)

        # 未知指令
        else:
            logger.warning(f"未知指令: {command}")
            return False
    
    def execute_commands(self, commands: Union[str, List[str]]) -> Dict[str, Any]:
        """
        执行多个指令
        
        参数:
            commands: AI的指令，可以是管道分隔的字符串或列表
        
        返回:
            执行结果统计
        """
        # 解析指令
        parsed_commands = self.parse_commands(commands)
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行所有指令
        results = {
            'total_commands': len(parsed_commands),
            'success_count': 0,
            'failed_commands': [],
            'execution_time': 0
        }
        
        logger.info(f"开始执行 {len(parsed_commands)} 条指令")
        
        for cmd in parsed_commands:
            success = self.execute_command(cmd)
            if success:
                results['success_count'] += 1
            else:
                results['failed_commands'].append(cmd)
            
            # 每个动作之间添加短暂延迟，避免操作过快
            time.sleep(0.05)
        
        # 计算执行时间
        results['execution_time'] = time.time() - start_time
        
        logger.info(f"指令执行完成: 成功 {results['success_count']}/{results['total_commands']}, 耗时 {results['execution_time']:.2f}秒")
        
        if results['failed_commands']:
            logger.warning(f"以下指令执行失败: {results['failed_commands']}")
        
        return results
    
    def set_move_step(self, step: int) -> None:
        """设置鼠标移动步长"""
        if step > 0:
            self.move_step = step
            logger.info(f"鼠标移动步长已设置为: {step}")
        else:
            logger.error(f"无效的移动步长: {step}")
    
    def shutdown(self) -> None:
        """关闭执行器"""
        self.mouse_controller.shutdown()
        logger.info("AI控制执行器已关闭")


def load_commands_from_file(file_path: str) -> Union[str, List[str], None]:
    """
    从文件加载指令
    
    参数:
        file_path: 包含指令的文件路径
    
    返回:
        加载的指令
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试解析为JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 如果不是有效的JSON，则作为字符串返回
                return content
    except Exception as e:
        logger.error(f"从文件加载指令失败: {e}")
        return None


def save_results_to_file(results: Dict[str, Any], file_path: str) -> bool:
    """
    保存执行结果到文件
    
    参数:
        results: 执行结果
        file_path: 保存结果的文件路径
    
    返回:
        保存是否成功
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存执行结果失败: {e}")
        return False


def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI控制指令执行器')
    parser.add_argument('--commands', type=str, help='AI的指令（管道分隔的字符串）')
    parser.add_argument('--commands_file', type=str, help='包含AI指令的文件路径')
    parser.add_argument('--move_step', type=int, default=50, help='鼠标移动步长')
    parser.add_argument('--output', type=str, help='保存执行结果的文件路径')
    
    args = parser.parse_args()
    
    # 初始化执行器
    executor = AIControlExecutor()
    
    try:
        # 设置移动步长
        executor.set_move_step(100)
        
        # 确定要执行的指令
        commands = None
        if args.commands:
            commands = args.commands
        elif args.commands_file:
            commands = load_commands_from_file(args.commands_file)
        else:
            parser.print_help()
            logger.error("必须提供指令或指令文件")
            return 1
        
        # 执行指令
        if commands:
            results = executor.execute_commands(commands)
            
            # 输出结果
            print(f"执行结果:")
            print(f"  总指令数: {results['total_commands']}")
            print(f"  成功指令数: {results['success_count']}")
            print(f"  失败指令数: {len(results['failed_commands'])}")
            print(f"  执行时间: {results['execution_time']:.2f}秒")
            
            if results['failed_commands']:
                print(f"  失败的指令: {results['failed_commands']}")
            
            # 保存结果到文件
            if args.output:
                save_results_to_file(results, args.output)
        
        return 0
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        return 1
    finally:
        executor.shutdown()


if __name__ == "__main__":
    sys.exit(main())