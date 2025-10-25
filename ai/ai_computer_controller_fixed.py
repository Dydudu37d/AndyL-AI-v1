#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 电脑控制系统 - 修复版本

该模块实现完整的AI操控电脑流程：
1. 使用screen_capture.py进行屏幕截图
2. 将截图提供给AI进行分析
3. 获取AI的控制指令
4. 使用ai_control_executor.py执行控制指令

此修复版本专注于改进命令解析逻辑，解决鼠标无法移动的问题。
"""

import os
import sys
import time
import logging
import json
import base64
from io import BytesIO
import requests
import numpy as np  # 添加numpy导入
import cv2  # 添加cv2导入
import re  # 显式导入re模块用于命令解析
from typing import Dict, Any, Optional, Callable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_computer_control.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AI_Computer_Controller")

# 确保可以导入所需模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from screen_capture import ScreenCapture
from ai_control_executor import AIControlExecutor


class AIComputerController:
    """AI电脑控制器 - 修复版本"""
    
    def __init__(self):
        """初始化控制器"""
        # 初始化屏幕捕获器
        self.screen_capturer = ScreenCapture()
        
        # 初始化指令执行器
        self.control_executor = AIControlExecutor()
        
        # 设置AI接口配置
        self.ai_api_url = ""  # AI API的URL
        self.api_key = ""     # API密钥
        self.ai_type = "api"  # AI类型: api, ollama, localai
        
        # Ollama配置
        self.ollama_host = "localhost"
        self.ollama_port = 11434
        self.ollama_model = "llama3.2:latest"
        
        # LocalAI配置
        self.localai_host = "localhost"
        self.localai_port = 8080
        self.localai_model = "gpt-3.5-turbo"
        
        # 截图相关配置
        self.screenshot_len = 0
        
        # 设置工作目录
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.screenshot_dir = os.path.join(self.work_dir, "screenshots")
        
        # 创建截图目录
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # 尝试从环境变量加载配置
        self._load_config_from_env()
        
        logger.info("AI电脑控制器已初始化")
        
    def _load_config_from_env(self):
        """从环境变量加载配置"""
        # Ollama配置
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        self.ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        
        # LocalAI配置
        self.localai_host = os.getenv("LOCALAI_HOST", "localhost")
        self.localai_port = int(os.getenv("LOCALAI_PORT", "8080"))
        self.localai_model = os.getenv("LOCALAI_MODEL", "gpt-3.5-turbo")
        
        # 设置默认AI类型为ollama，因为这是最常用的本地AI选项
        env_ai_type = os.getenv("AI_TYPE", "ollama")
        if env_ai_type in ["ollama", "localai", "api"]:
            self.ai_type = env_ai_type
        
        # 如果是API类型，加载API配置
        if self.ai_type == "api":
            self.ai_api_url = os.getenv("AI_API_URL", "")
            self.api_key = os.getenv("AI_API_KEY", "")
    
    def set_ai_api_config(self, api_url: str, api_key: str) -> None:
        """设置AI API配置"""
        self.ai_type = "api"
        self.ai_api_url = api_url
        self.api_key = api_key
        logger.info(f"AI API配置已设置: URL={api_url}")
        
    def set_ollama_config(self, host: str = "localhost", port: int = 11434, model: str = "llama3.2:latest") -> None:
        """设置Ollama配置"""
        self.ai_type = "ollama"
        self.ollama_host = host
        self.ollama_port = port
        self.ollama_model = model
        logger.info(f"Ollama配置已设置: {host}:{port}, 模型={model}")
        
    def set_localai_config(self, host: str = "localhost", port: int = 8080, model: str = "gpt-3.5-turbo") -> None:
        """设置LocalAI配置"""
        self.ai_type = "localai"
        self.localai_host = host
        self.localai_port = port
        self.localai_model = model
        logger.info(f"LocalAI配置已设置: {host}:{port}, 模型={model}")
    
    def capture_screen(self, region=None) -> str:
        """
        捕获屏幕并保存截图
        
        参数:
            region: 捕获区域 (x, y, width, height)，None表示全屏
        
        返回:
            保存的截图文件路径
        """
        try:
            # 如果有指定区域，先设置给screen_capturer
            if region:
                self.screen_capturer.region = region
            
            # 捕获屏幕（不需要传递region参数）
            screenshot = self.screen_capturer.capture_screen()
            
            if screenshot is None:
                logger.error("屏幕捕获失败")
                return ""
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
            
            # 保存截图
            # 判断screenshot类型，使用对应的保存方法
            if hasattr(screenshot, 'save'):
                # PIL Image对象直接调用save
                screenshot.save(screenshot_path)
            elif isinstance(screenshot, np.ndarray):
                # numpy数组使用cv2.imwrite保存
                # 确保图像格式正确
                if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
                    # BGR格式转换为RGB
                    cv2.imwrite(screenshot_path, cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                else:
                    cv2.imwrite(screenshot_path, screenshot)
            logger.info(f"屏幕截图已保存: {screenshot_path}")
            
            return screenshot_path
        except Exception as e:
            logger.error(f"捕获屏幕时发生错误: {e}")
            return ""
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图像编码为base64字符串
        
        参数:
            image_path: 图像文件路径
        
        返回:
            base64编码的图像字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图像为base64时发生错误: {e}")
            return ""
    
    def send_to_ai(self, screenshot_path: str, prompt: str = "") -> Optional[str]:
        """将截图和提示发送给AI，获取控制指令
        
        参数:
            screenshot_path: 截图文件路径
            prompt: 给AI的提示信息
        
        返回:
            AI的控制指令
        """
        try:
            # 根据AI类型选择不同的调用方法
            if self.ai_type == "api":
                return self._send_to_api(screenshot_path, prompt)
            elif self.ai_type == "ollama":
                return self._send_to_ollama(screenshot_path, prompt)
            elif self.ai_type == "localai":
                return self._send_to_localai(screenshot_path, prompt)
            else:
                logger.error(f"不支持的AI类型: {self.ai_type}")
                return None
        except Exception as e:
            logger.error(f"发送请求到AI时发生错误: {e}")
            return None
    
    def _send_to_api(self, screenshot_path: str, prompt: str = "") -> Optional[str]:
        """发送请求到API类型的AI服务"""
        if not self.ai_api_url:
            logger.error("AI API URL未设置")
            return None
        
        # 将图像编码为base64
        image_base64 = self.encode_image_to_base64(screenshot_path)
        
        if not image_base64:
            return None
        
        # 构建请求数据
        payload = {
            "image": image_base64,
            "prompt": prompt,
            "format": "pipe"  # 可以是 'pipe' 或 'list'
        }
        
        # 发送请求到AI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"发送请求到AI API: {self.ai_api_url}")
        response = requests.post(
            self.ai_api_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            
            # 假设AI返回的指令在 'commands' 字段中
            if "commands" in result:
                commands = result["commands"]
                logger.info(f"成功获取AI指令: {commands}")
                return commands
            else:
                logger.error(f"AI响应中不包含commands字段: {result}")
                return None
        else:
            logger.error(f"AI API请求失败: {response.status_code}, {response.text}")
            return None
    
    def _send_to_ollama(self, screenshot_path: str, prompt: str = "") -> Optional[str]:
        """发送请求到Ollama本地AI服务 - 改进版本"""
        try:
            # 将图像编码为base64
            image_base64 = self.encode_image_to_base64(screenshot_path)
            
            if not image_base64:
                return None
            
            # 构建Ollama请求数据
            payload = {
                "model": self.ollama_model,
                "prompt": self._build_ollama_prompt(prompt),
                "images": [image_base64],
                "stream": False
            }
            
            # 发送请求到Ollama
            api_url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
            logger.info(f"发送请求到Ollama: {api_url}, 模型={self.ollama_model}")
            
            response = requests.post(
                api_url,
                json=payload,
                timeout=60
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                
                # 提取Ollama的响应内容
                if "response" in result:
                    ai_response = result["response"]
                    logger.info(f"成功获取Ollama响应: {ai_response}")
                    # 解析响应以获取命令
                    commands = self._parse_local_ai_response(ai_response)
                    logger.info(f"解析出的命令: {commands}")
                    return commands
                else:
                    logger.error(f"Ollama响应格式不匹配: {result}")
                    return None
            else:
                logger.error(f"Ollama请求失败: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            logger.error(f"发送请求到Ollama时发生错误: {e}")
            return None
    
    def _send_to_localai(self, screenshot_path: str, prompt: str = "") -> Optional[str]:
        """发送请求到LocalAI本地AI服务"""
        try:
            # 将图像编码为base64
            image_base64 = self.encode_image_to_base64(screenshot_path)
            
            if not image_base64:
                return None
            
            # 构建LocalAI请求数据
            messages = [
                {
                    "role": "system",
                    "content": "你是一个AI助手，可以根据屏幕内容生成控制电脑的指令。请根据用户的提示和屏幕截图分析，返回适当的控制指令。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_localai_prompt(prompt)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            payload = {
                "model": self.localai_model,
                "messages": messages,
                "max_tokens": 500
            }
            
            # 发送请求到LocalAI
            api_url = f"http://{self.localai_host}:{self.localai_port}/v1/chat/completions"
            logger.info(f"发送请求到LocalAI: {api_url}, 模型={self.localai_model}")
            
            response = requests.post(
                api_url,
                json=payload,
                timeout=60
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                
                # 提取LocalAI的响应内容
                if "choices" in result and len(result["choices"]) > 0:
                    ai_response = result["choices"][0]["message"]["content"]
                    logger.info(f"成功获取LocalAI响应: {ai_response}")
                    # 解析响应以获取命令
                    commands = self._parse_local_ai_response(ai_response)
                    logger.info(f"解析出的命令: {commands}")
                    return commands
                else:
                    logger.error(f"LocalAI响应格式不匹配: {result}")
                    return None
            else:
                logger.error(f"LocalAI请求失败: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            logger.error(f"发送请求到LocalAI时发生错误: {e}")
            return None
    
    def _build_ollama_prompt(self, original_prompt: str) -> str:
        """构建发送给Ollama的提示信息 - 改进版本"""
        system_prompt = "你是一个AI助手，可以根据屏幕内容生成控制电脑的指令。"
        system_prompt += "请分析屏幕截图和用户提示，然后返回一系列控制指令。"
        system_prompt += "指令格式必须是用|分隔的命令列表，例如: 'm_up|m_left|m_click|t_hello'"
        system_prompt += "请只返回命令列表，不要包含任何额外的解释文本。"
        system_prompt += "\n\n可用命令:\n"
        system_prompt += "- m_up/m_down/m_left/m_right: 移动鼠标\n"
        system_prompt += "- m_click/m_right_click/m_double_click: 鼠标点击\n"
        system_prompt += "- t_text: 输入文本\n"
        system_prompt += "- k_key: 按键\n"
        
        return f"{system_prompt}\n\n用户提示: {original_prompt}\n请仅返回命令列表："
    
    def _build_localai_prompt(self, original_prompt: str) -> str:
        """构建发送给LocalAI的提示信息"""
        return f"请根据屏幕截图和以下提示，返回一系列控制电脑的指令。{original_prompt}\n\n指令格式必须是用|分隔的命令列表，例如: 'm_up|m_left|m_click|t_hello'\n请只返回命令列表，不要包含任何额外的解释文本。"
    
    def _parse_local_ai_response(self, ai_response: str) -> str:
        """解析本地AI的响应以提取命令 - 增强版本"""
        import re
        
        logger.info(f"开始解析AI响应: {ai_response}")
        
        # 预处理：去除多余的空格和换行符
        processed_response = ai_response.strip()
        
        # 尝试1: 提取代码块中的命令
        code_match = re.search(r'```(?:\w*)\n(.*?)\n```', processed_response, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
            # 检查代码块内容是否包含命令
            if '|' in code_content:
                logger.info(f"从代码块中提取命令: {code_content}")
                return code_content
        
        # 尝试2: 提取明确的命令列表 - 多行匹配
        command_pattern = r'([a-z_]+\|(?:[a-z_]+\|)*[a-z_]+)'  # 匹配m_up|m_down|...格式
        command_match = re.search(command_pattern, processed_response, re.MULTILINE | re.IGNORECASE)
        if command_match:
            logger.info(f"提取命令列表: {command_match.group(1)}")
            return command_match.group(1)
        
        # 尝试3: 提取单行中的命令列表（增强版）
        single_line_match = re.search(r'(?:["\'])?([a-z_]+(?:\|[a-z_]+)+)(?:["\'])?', processed_response, re.IGNORECASE)
        if single_line_match:
            logger.info(f"从单行中提取命令: {single_line_match.group(1)}")
            return single_line_match.group(1)
        
        # 尝试4: 检查是否有特殊的命令格式（m_、k_、t_开头）
        t_command_match = re.search(r't_\w+', processed_response)
        k_command_match = re.search(r'k_\w+', processed_response)
        m_command_match = re.search(r'm_\w+', processed_response)
        
        if t_command_match or k_command_match or m_command_match:
            # 提取所有命令并组合
            commands = []
            if t_command_match:
                commands.append(t_command_match.group(0))
            if k_command_match:
                commands.append(k_command_match.group(0))
            if m_command_match:
                commands.append(m_command_match.group(0))
            
            if commands:
                combined_commands = '|'.join(commands)
                logger.info(f"组合特殊命令: {combined_commands}")
                return combined_commands
        
        # 尝试5: 查找所有以|分隔的单词
        pipe_separated_matches = re.findall(r'\b(?:m|k|t)_\w+\b', processed_response)
        if pipe_separated_matches:
            combined = '|'.join(pipe_separated_matches)
            logger.info(f"从响应中提取所有命令: {combined}")
            return combined
        
        # 尝试6: 检查是否只包含命令列表
        if re.match(r'^(?:[a-z_]+\|)*[a-z_]+$', processed_response):
            logger.info(f"响应本身就是命令列表: {processed_response}")
            return processed_response
        
        # 如果以上所有尝试都失败，返回空字符串而不是整个响应
        # 这样执行器不会尝试执行无法识别的命令
        logger.warning(f"未能从AI响应中提取有效命令: {ai_response}")
        return ""
    
    def mock_ai_response(self, screenshot_path: str, prompt: str = "") -> str:
        """
        模拟AI的响应（用于测试）
        
        参数:
            screenshot_path: 截图文件路径
            prompt: 给AI的提示信息
        
        返回:
            模拟的AI控制指令
        """
        logger.info(f"使用模拟AI响应，截图: {screenshot_path}, 提示: {prompt}")
        
        # 根据不同情况返回不同的模拟指令
        prompt_lower = prompt.lower()
        
        if "test_move" in prompt_lower:
            # 模拟移动和点击
            return "m_up|m_up|m_up|m_left|m_left|m_click"
        elif "test_type" in prompt_lower:
            # 模拟输入文本
            return "h|e|l|l|o|space|w|o|r|l|d|"
        elif "test_combination" in prompt_lower:
            # 模拟组合键
            return "ctrl+a|ctrl+c"
        elif "探索桌面" in prompt_lower:
            # 模拟探索桌面环境的操作
            return "m_right|m_right|m_down|m_down|m_click|m_up|m_up|m_left|m_left|m_click"
        elif "随机鼠标" in prompt_lower:
            # 模拟随机鼠标移动测试
            return "m_right|m_down|m_left|m_up|m_click|m_down|m_right|m_up|m_left|m_click"
        elif "模拟用户" in prompt_lower:
            # 模拟用户操作
            return "m_right|m_right|m_down|m_down|m_click|w|i|n|d|o|w|s|space|e|x|p|l|o|r|e"
        elif "混合控制" in prompt_lower:
            # 混合控制指令
            return "ctrl|shift|escape|m_right|m_down|m_click|c|o|n|t|r|o|l|space|m|o|d|e"
        elif "多样化" in prompt_lower:
            # 多样化电脑控制
            return "m_up|m_up|m_left|m_left|m_click|ctrl|c|ctrl|v|enter"
        else:
            # 默认情况下，返回随机选择的指令，避免重复
            import random
            command_sets = [
                "m_right|m_right|m_down|m_down|m_click",
                "m_up|m_up|m_left|m_left|m_click",
                "m_down|m_down|m_right|m_right|m_click",
                "m_left|m_left|m_up|m_up|m_click",
                "w|o|r|k|space|o|n|space|p|r|o|j|e|c|t"
            ]
            return random.choice(command_sets)
    
    def execute_ai_commands(self, commands: str) -> Dict[str, Any]:
        """
        执行AI的控制指令
        
        参数:
            commands: AI的控制指令
        
        返回:
            执行结果统计
        """
        if not commands:
            logger.warning("没有要执行的命令")
            return {"success_count": 0, "total_commands": 0, "failed_commands": []}
            
        logger.info(f"准备执行AI命令: {commands}")
        return self.control_executor.execute_commands(commands)
    
    def run_single_cycle(self, prompt: str = "", use_mock: bool = True) -> Dict[str, Any]:
        """
        运行单个控制循环
        
        参数:
            prompt: 给AI的提示信息
            use_mock: 是否使用模拟AI响应
        
        返回:
            循环执行结果
        """
        cycle_result = {
            "success": False,
            "screenshot_path": "",
            "commands": "",
            "execution_results": {},
            "error": ""
        }
        
        try:
            
            self.screenshot_len += 1
            
            # 1. 捕获屏幕
            screenshot_path = self.capture_screen()
            if not screenshot_path:
                cycle_result["error"] = "屏幕捕获失败"
                return cycle_result
            
            cycle_result["screenshot_path"] = screenshot_path
            
            # 2. 获取AI指令
            if use_mock:
                commands = self.mock_ai_response(screenshot_path, prompt)
            else:
                commands = self.send_to_ai(screenshot_path, prompt)
            
            if not commands:
                cycle_result["error"] = "获取AI指令失败或指令为空"
                return cycle_result
            
            cycle_result["commands"] = commands
            
            # 3. 执行指令
            execution_results = self.execute_ai_commands(commands)
            cycle_result["execution_results"] = execution_results
            
            # 4. 设置成功标志
            cycle_result["success"] = True
            
            logger.info(f"控制循环执行成功: 命令={commands}, 执行结果={execution_results}")
            if self.screenshot_len == 3:
                os.remove(screenshot_path)
                self.screenshot_len = 0
        except Exception as e:
            error_msg = f"控制循环执行失败: {e}"
            logger.error(error_msg)
            cycle_result["error"] = error_msg
        
        return cycle_result
    
    def run_continuous(self, prompt: str = "", cycles: int = 10, interval: float = 2.0, use_mock: bool = True) -> list[Dict[str, Any]]:
        """
        连续运行多个控制循环
        
        参数:
            prompt: 给AI的提示信息
            cycles: 运行的循环次数
            interval: 循环间隔时间（秒）
            use_mock: 是否使用模拟AI响应
        
        返回:
            所有循环的执行结果
        """
        all_results = []
        
        logger.info(f"开始连续控制循环: 次数={cycles}, 间隔={interval}秒")
        
        try:
            for i in range(cycles):
                logger.info(f"执行控制循环 {i+1}/{cycles}")
                
                # 运行单个循环
                result = self.run_single_cycle(prompt, use_mock)
                all_results.append(result)
                
                # 等待指定的间隔时间
                if i < cycles - 1:
                    time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("用户中断了连续控制循环")
        except Exception as e:
            logger.error(f"连续控制循环执行失败: {e}")
        
        logger.info(f"连续控制循环完成，共执行 {len(all_results)} 个循环")
        
        # 统计成功次数
        success_count = sum(1 for r in all_results if r["success"])
        logger.info(f"连续控制循环统计: 成功 {success_count}/{len(all_results)} 个循环")
        
        return all_results
    
    def shutdown(self) -> None:
        """关闭控制器"""
        # 关闭屏幕捕获器
        self.screen_capturer.stop_capturing()
        
        # 关闭控制执行器
        self.control_executor.shutdown()
        
        logger.info("AI电脑控制器已关闭")


class CustomAIAgent:
    """自定义AI代理（用于演示）"""
    
    def __init__(self):
        """初始化自定义AI代理"""
        self.screen_analyzer = None  # 可以集成图像分析模型
        logger.info("自定义AI代理已初始化")
    
    def analyze_screenshot(self, screenshot_path: str) -> Dict[str, Any]:
        """分析截图并生成操作指令"""
        # 这里只是一个示例实现
        # 实际应用中，可以使用OCR、对象检测等技术分析屏幕内容
        
        logger.info(f"分析截图: {screenshot_path}")
        
        # 模拟分析结果
        analysis = {
            "detected_elements": [],
            "recommended_actions": "m_up|m_up|m_left|m_click"
        }
        
        return analysis


# 演示函数


def demo_single_cycle():
    """演示单个控制循环"""
    print("======= 演示单个控制循环 =======")
    
    controller = AIComputerController()
    
    try:
        # 运行单个控制循环
        result = controller.run_single_cycle("这是一个测试，模拟AI控制电脑", use_mock=True)
        
        # 打印结果
        print(f"控制循环结果: {result}")
    finally:
        controller.shutdown()


def demo_continuous_control():
    """演示连续控制"""
    print("======= 演示连续控制 =======")
    
    controller = AIComputerController()
    
    try:
        # 连续运行5个控制循环，间隔1秒
        results = controller.run_continuous("连续控制测试", cycles=5, interval=1.0, use_mock=True)
        
        # 打印结果统计
        success_count = sum(1 for r in results if r["success"])
        print(f"连续控制结果: 成功 {success_count}/{len(results)} 个循环")
    finally:
        controller.shutdown()


def demo_custom_workflow():
    """演示自定义工作流程"""
    print("======= 演示自定义工作流程 =======")
    
    # 初始化组件
    screen_capturer = ScreenCapture()
    executor = AIControlExecutor()
    custom_ai = CustomAIAgent()
    
    try:
        # 1. 捕获屏幕
        screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots", "custom_workflow.png")
        screenshot = screen_capturer.capture_screen()
        screenshot.save(screenshot_path)
        
        # 2. 使用自定义AI分析屏幕
        analysis = custom_ai.analyze_screenshot(screenshot_path)
        
        # 3. 执行AI建议的操作
        results = executor.execute_commands(analysis["recommended_actions"])
        
        print(f"自定义工作流程结果: {results}")
    finally:
        # 清理资源
        executor.shutdown()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI电脑控制系统')
    parser.add_argument('--demo', type=str, choices=['single', 'continuous', 'custom'], 
                        help='运行演示模式: single(单个循环), continuous(连续循环), custom(自定义工作流程)')
    parser.add_argument('--cycles', type=int, default=5, help='连续演示的循环次数')
    parser.add_argument('--interval', type=float, default=1.0, help='连续演示的间隔时间(秒)')
    parser.add_argument('--use-mock', type=bool, default=True, help='是否使用模拟AI响应')
    
    args = parser.parse_args()
    
    # 运行相应的演示
    if args.demo == 'single':
        demo_single_cycle()
    elif args.demo == 'continuous':
        demo_continuous_control()
    elif args.demo == 'custom':
        demo_custom_workflow()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()