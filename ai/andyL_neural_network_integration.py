# AndyL神经网络与AI系统集成模块

import numpy as np
import cv2
import logging
import os
import time
from typing import Dict, List, Optional, Callable, Union
import random
import torch
import torch.nn as nn

# 导入我们的模型加载工具
from load_andyL_model import load_andyL_model, AndyLModelLoader

# 配置日志
logger = logging.getLogger("AndyLNeuralNetwork")
logger.setLevel(logging.INFO)

class AndyLNeuralNetwork:
    """
    AndyL神经网络核心类
    负责加载、管理和运行神经网络模型，处理屏幕截图等视觉输入
    支持实际的PyTorch和ONNX模型，以及模拟模式
    """
    
    def __init__(self, use_real_model: bool = True):
        # 模型相关属性
        self.model = "simulated"  # 默认使用模拟模型
        self.model_type = "simulation"  # 模拟模式
        self.model_path = None
        self.metadata = None  # 模型元数据
        self.use_real_model = use_real_model  # 是否使用实际模型
        
        # 输入预处理配置
        self.input_shape = (84, 84, 3)  # 默认输入形状
        self.normalization = True       # 是否归一化输入
        
        # 推理配置
        self.use_gpu = torch.cuda.is_available() # 是否使用GPU
        self.inference_time = 0         # 记录推理时间
        
        # 创建模型加载器
        self.model_loader = AndyLModelLoader()
        
        # 初始化
        if self.use_real_model:
            logger.info("AndyL神经网络已初始化，准备加载实际模型")
        else:
            logger.info("使用模拟神经网络模型")
    
    # _initialize_environment方法已移除，因为模拟模式不需要初始化深度学习环境
    
    def load_model(self, model_path: str, model_class: Optional[torch.nn.Module] = None):
        """
        加载预训练的AndyL神经网络模型（支持实际加载和模拟加载）
        
        参数:
            model_path: 模型文件路径
            model_class: 可选的模型类，用于加载自定义pth模型结构
        
        返回:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            self.model_path = model_path
            
            start_time = time.time()
            
            if self.use_real_model:
                # 实际加载模型
                self.model, self.metadata, self.model_type = load_andyL_model(model_path, model_class)
                
                if self.model is None:
                    logger.error(f"实际模型加载失败: {model_path}")
                    # 回退到模拟模式
                    self.model = "simulated"
                    self.model_type = "simulation"
                    logger.warning("回退到模拟模式")
                    return False
                
                logger.info(f"成功加载{self.model_type.upper()}模型: {os.path.basename(model_path)}")
                
                # 检查是否支持GPU
                if torch.cuda.is_available():
                    self.use_gpu = True
                    # 如果是PyTorch模型，移至GPU
                    if self.model_type == 'pth' and isinstance(self.model, torch.nn.Module):
                        self.model.to('cuda')
                    logger.info("已启用GPU加速")
            else:
                # 模拟模型加载
                _, ext = os.path.splitext(model_path.lower())
                if ext == '.pth':
                    self.model_type = 'pytorch'
                elif ext == '.onnx':
                    self.model_type = 'onnx'
                else:
                    self.model_type = 'unknown'
                
                self.model = "simulated_" + self.model_type
                logger.info(f"模拟加载{self.model_type}模型: {model_path}")
            
            load_time = time.time() - start_time
            logger.info(f"模型加载时间: {load_time:.2f}秒")
            
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """
        预处理输入图像以适应神经网络
        
        参数:
            image: 原始图像数组
        
        返回:
            预处理后的图像数组
        """
        try:
            # 调整大小以匹配模型输入
            h, w = self.input_shape[:2]
            resized = cv2.resize(image, (w, h))
            
            # 转换为RGB格式（如果需要）
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
            
            # 添加批次维度
            input_tensor = np.expand_dims(resized, axis=0)
            
            # 归一化
            if self.normalization:
                input_tensor = input_tensor.astype(np.float32) / 255.0
            
            return input_tensor
        except Exception as e:
            logger.error(f"预处理输入失败: {e}")
            return None
    
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用模型进行预测（支持实际模型和模拟模式）
        
        参数:
            input_data: 预处理后的输入数据
        
        返回:
            预测结果字典
        """
        try:
            if self.model is None:
                logger.error("模型未加载")
                return {"error": "模型未加载"}
            
            # 记录推理开始时间
            start_time = time.time()
            
            # 初始化结果字典
            result = {}
            
            if self.use_real_model and self.model_type in ['pth', 'onnx']:
                # 使用实际模型进行推理
                try:
                    if self.model_type == 'pth' and isinstance(self.model, torch.nn.Module):
                        # PyTorch模型推理
                        with torch.no_grad():
                            # 确保输入是tensor
                            if not isinstance(input_data, torch.Tensor):
                                input_tensor = torch.tensor(input_data)
                                # 如果启用了GPU，移至GPU
                                if self.use_gpu:
                                    input_tensor = input_tensor.to('cuda')
                            else:
                                input_tensor = input_data
                            
                            # 执行推理
                            model_output = self.model(input_tensor)
                            
                            # 转换为numpy数组
                            if isinstance(model_output, torch.Tensor):
                                predictions = model_output.cpu().numpy()
                            else:
                                predictions = np.array(model_output)
                    
                    elif self.model_type == 'onnx' and hasattr(self.model, 'run'):
                        # ONNX模型推理
                        # 获取输入名称
                        input_name = self.model.get_inputs()[0].name
                        
                        # 确保输入是numpy数组
                        if isinstance(input_data, torch.Tensor):
                            input_data_np = input_data.cpu().numpy()
                        else:
                            input_data_np = np.array(input_data)
                        
                        # 执行推理
                        inputs = {input_name: input_data_np}
                        outputs = self.model.run(None, inputs)
                        
                        # 假设第一个输出是主要预测结果
                        predictions = outputs[0]
                    
                    else:
                        logger.error(f"不支持的模型类型: {self.model_type}")
                        # 回退到模拟结果
                        batch_size = input_data.shape[0] if isinstance(input_data, np.ndarray) else 1
                        predictions = np.random.random((batch_size, 10))
                    
                    result["main_prediction"] = predictions
                    
                except Exception as e:
                    logger.error(f"实际模型推理失败: {e}")
                    # 回退到模拟结果
                    batch_size = input_data.shape[0] if isinstance(input_data, np.ndarray) else 1
                    predictions = np.random.random((batch_size, 10))
                    result["main_prediction"] = predictions
                    result["simulation_fallback"] = True
            else:
                # 模拟推理结果
                batch_size = input_data.shape[0] if isinstance(input_data, np.ndarray) else 1
                predictions = np.random.random((batch_size, 10))  # 假设有10个类别
                result["main_prediction"] = predictions
                result["simulation_mode"] = True
            
            # 记录推理时间
            self.inference_time = time.time() - start_time
            logger.debug(f"推理时间: {self.inference_time:.4f}秒")
            
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {"error": str(e)}
    
    def preprocess_image(self, screen_image: np.ndarray) -> np.ndarray:
        """
        预处理屏幕图像
        
        参数:
            screen_image: 原始屏幕图像 (BGR格式)
            
        返回:
            预处理后的图像数据
        """
        try:
            # 调整图像大小
            resized = cv2.resize(screen_image, (self.input_shape[1], self.input_shape[0]))
            
            # 如果需要归一化
            if self.normalization:
                # 将像素值归一化到0-1范围
                normalized = resized.astype(np.float32) / 255.0
                return normalized
            else:
                return resized
                
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            return None
            
    def process_screen_image(self, screenshot: np.ndarray) -> Dict[str, np.ndarray]:
        """
        处理屏幕图像并生成控制命令（支持实际模型和模拟模式）
        
        参数:
            screenshot: 原始屏幕截图
        
        返回:
            Dict: 包含预测结果和控制命令的字典
        """
        try:
            # 预处理图像
            processed_image = self.preprocess_image(screenshot)
            if processed_image is None:
                return {"error": "图像预处理失败"}
            
            # 添加批次维度
            input_data = np.expand_dims(processed_image, axis=0)
            
            # 使用实际的predict方法进行推理
            predictions = self.predict(input_data)
            
            # 检查是否有错误
            if "error" in predictions:
                return {"error": predictions["error"]}
            
            # 解析预测结果，生成控制命令
            control_commands = self._parse_predictions(predictions)
            
            # 构建返回结果
            result = {
                "predictions": predictions,
                "control_commands": control_commands,
                "processing_time": self.inference_time,
                "original_shape": screenshot.shape,
                "processed_shape": processed_image.shape
            }
            
            return result
        except Exception as e:
            logger.error(f"处理屏幕图像失败: {e}")
            return {"error": str(e)}
    
    def _parse_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        解析预测结果，生成控制命令（模拟实现）
        
        参数:
            predictions: 模型预测结果
            
        返回:
            Dict: 控制命令字典
        """
        control_commands = {}
        
        try:
            # 随机选择一个动作类型
            action_types = ["mouse_move", "left_click", "right_click", "key_press", "mouse_scroll", "none"]
            action_type = random.choice(action_types)
            
            # 根据动作类型生成不同的控制命令
            if action_type == "mouse_move":
                control_commands["action"] = "mouse_move"
                control_commands["x"] = float(random.randint(0, 1920))
                control_commands["y"] = float(random.randint(0, 1080))
            elif action_type == "left_click":
                control_commands["action"] = "left_click"
            elif action_type == "right_click":
                control_commands["action"] = "right_click"
            elif action_type == "key_press":
                control_commands["action"] = "key_press"
                control_commands["key"] = "space"  # 示例按键
            elif action_type == "mouse_scroll":
                control_commands["action"] = "mouse_scroll"
                control_commands["delta"] = float(random.randint(-100, 100))
            elif action_type == "none":
                control_commands["action"] = "none"
            
            # 随机置信度分数
            control_commands["confidence"] = float(random.uniform(0.5, 1.0))
            
        except Exception as e:
            logger.error(f"解析预测结果时出错: {str(e)}")
            
        return control_commands

class AndyLNNIntegration:
    """
    AndyL神经网络与AI系统集成类
    负责将神经网络与现有AndyL AI系统无缝集成
    """
    
    def __init__(self, ai_brain=None, mouse_controller=None):
        # 核心组件引用
        self.ai_brain = ai_brain
        self.mouse_controller = mouse_controller
        
        # 神经网络实例
        self.neural_network = AndyLNeuralNetwork()
        
        # 集成状态
        self.is_integrated = False
        self.integration_callbacks = []
    
    def integrate(self, model_path: str = None, model_type: str = "tensorflow"):
        """
        集成神经网络到AndyL AI系统
        
        参数:
            model_path: 可选的模型路径
            model_type: 模型类型
        
        返回:
            bool: 集成是否成功
        """
        try:
            # 如果提供了模型路径，加载模型
            if model_path:
                if not self.neural_network.load_model(model_path, model_type):
                    return False
            
            # 注册集成回调
            self.is_integrated = True
            logger.info("AndyL神经网络与AI系统集成成功")
            
            # 调用所有注册的回调
            for callback in self.integration_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"集成回调执行失败: {e}")
            
            return True
        except Exception as e:
            logger.error(f"集成失败: {e}")
            return False
    
    def register_integration_callback(self, callback: Callable):
        """
        注册集成完成后的回调函数
        
        参数:
            callback: 回调函数
        """
        self.integration_callbacks.append(callback)
        
        # 如果已经集成，立即调用回调
        if self.is_integrated:
            try:
                callback()
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def process_screen_and_control(self, screenshot: np.ndarray) -> Dict:
        """
        处理屏幕截图并生成控制命令
        
        参数:
            screenshot: 屏幕截图
        
        返回:
            包含处理结果和控制命令的字典
        """
        try:
            # 使用神经网络处理屏幕
            nn_result = self.neural_network.process_screen_image(screenshot)
            
            # 如果有错误，返回错误信息
            if "error" in nn_result:
                return {"success": False, "error": nn_result["error"]}
            
            # 如果有AI大脑，可以让AI进一步分析结果并生成控制命令
            control_result = {"success": True, "nn_result": nn_result}
            
            if self.ai_brain:
                # 将神经网络结果转换为文本描述
                nn_result_text = self._convert_nn_result_to_text(nn_result)
                
                # 创建提示词让AI生成控制命令
                prompt = self._create_ai_prompt_for_control(nn_result_text, screenshot)
                
                # 获取AI响应
                ai_response = self.ai_brain.process_text(prompt)
                control_result["ai_response"] = ai_response
                
                # 如果有鼠标控制器，执行控制命令
                if self.mouse_controller:
                    executed_commands = self._execute_ai_commands(ai_response)
                    control_result["executed_commands"] = executed_commands
            
            return control_result
        except Exception as e:
            logger.error(f"处理屏幕和控制失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _convert_nn_result_to_text(self, nn_result: Dict) -> str:
        """将神经网络结果转换为文本描述"""
        try:
            description = "神经网络分析结果：\n"
            
            # 添加基本信息
            description += f"- 原始图像尺寸: {nn_result['original_shape']}\n"
            description += f"- 处理后尺寸: {nn_result['processed_shape']}\n"
            description += f"- 推理时间: {nn_result['inference_time']:.4f}秒\n"
            
            # 添加预测结果（根据实际模型输出格式调整）
            if "main_prediction" in nn_result:
                pred = nn_result["main_prediction"]
                if isinstance(pred, np.ndarray) and pred.ndim == 2:
                    # 假设是分类结果
                    class_idx = np.argmax(pred[0])
                    confidence = np.max(pred[0])
                    description += f"- 主要预测类别: {class_idx}, 置信度: {confidence:.2f}\n"
                else:
                    description += f"- 预测数据形状: {pred.shape}\n"
            
            return description
        except Exception as e:
            logger.error(f"转换神经网络结果失败: {e}")
            return "无法解析神经网络结果"
    
    def _create_ai_prompt_for_control(self, nn_result_text: str, screenshot: np.ndarray) -> str:
        """创建AI控制提示词"""
        height, width = screenshot.shape[:2]
        
        prompt = f"""
你正在控制一台电脑，以下是当前状态：
{nn_result_text}

根据神经网络分析结果和屏幕状态，决定是否需要执行控制操作。
如果需要控制，请生成控制命令；如果不需要控制，请返回'none'。

控制命令格式应为以下之一：
1. move_mouse(x, y, duration) - 移动鼠标到指定坐标
2. click(button, count) - 点击鼠标
3. scroll(dx, dy) - 滚动鼠标滚轮
4. type_string(text) - 输入文本
5. press_key(key) - 按下键盘按键

请只返回控制命令，不要添加其他任何文字说明。
        """
        
        return prompt
    
    def _execute_ai_commands(self, ai_response: str) -> List[Dict]:
        """执行AI生成的控制命令"""
        import re
        executed_commands = []
        
        try:
            if not ai_response or ai_response.strip().lower() == 'none':
                return executed_commands
            
            # 解析move_mouse命令
            move_match = re.search(r'move_mouse\((\d+),\s*(\d+)(?:,\s*(\d+\.\d+))?\)', ai_response)
            if move_match:
                x = int(move_match.group(1))
                y = int(move_match.group(2))
                duration = float(move_match.group(3)) if move_match.group(3) else 0.2
                self.mouse_controller.move_mouse(x, y, duration)
                executed_commands.append({"type": "move_mouse", "x": x, "y": y})
                
            # 解析click命令
            click_match = re.search(r'click\((\'|"|)(left|right|middle)\1(?:,\s*(\d+))?\)', ai_response)
            if click_match:
                button = click_match.group(2)
                count = int(click_match.group(3)) if click_match.group(3) else 1
                self.mouse_controller.click(button=button, count=count)
                executed_commands.append({"type": "click", "button": button})
                
            # 解析scroll命令
            scroll_match = re.search(r'scroll\((-?\d+),\s*(-?\d+)\)', ai_response)
            if scroll_match:
                dx = int(scroll_match.group(1))
                dy = int(scroll_match.group(2))
                self.mouse_controller.scroll(dx, dy)
                executed_commands.append({"type": "scroll", "dx": dx, "dy": dy})
                
        except Exception as e:
            logger.error(f"执行AI命令失败: {e}")
        
        return executed_commands

# 创建单例实例
_andyL_nn_instance = None

# 工厂函数：获取AndyL神经网络集成实例
def get_andyL_neural_network() -> AndyLNNIntegration:
    """获取AndyL神经网络集成实例（单例模式）"""
    global _andyL_nn_instance
    if _andyL_nn_instance is None:
        _andyL_nn_instance = AndyLNNIntegration()
    return _andyL_nn_instance

# 主程序入口：演示集成用法
if __name__ == "__main__":
    # 配置详细日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建集成实例
    nn_integration = get_andyL_neural_network()
    
    print("=== AndyL神经网络集成演示 ===")
    print("1. 初始化神经网络环境")
    
    # 模拟集成到现有系统
    class MockAIBrain:
        def process_text(self, text, use_stream=False):
            print(f"模拟AI处理文本: {text[:100]}...")
            # 返回模拟命令
            return "move_mouse(500, 500, 0.2)"
    
    class MockMouseController:
        def move_mouse(self, x, y, duration=0.2):
            print(f"模拟移动鼠标到 ({x}, {y})")
        def click(self, button="left", count=1):
            print(f"模拟点击鼠标: {button}")
        def scroll(self, dx, dy):
            print(f"模拟滚动鼠标: dx={dx}, dy={dy}")
    
    # 设置模拟组件
    nn_integration.ai_brain = MockAIBrain()
    nn_integration.mouse_controller = MockMouseController()
    
    print("2. 集成神经网络")
    # 在实际使用中，这里应该提供真实的模型路径
    # nn_integration.integrate("path/to/your/model")
    
    # 由于没有实际模型，我们手动设置为已集成
    nn_integration.is_integrated = True
    
    print("3. 模拟处理屏幕并控制")
    # 创建一个模拟屏幕截图
    mock_screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # 处理屏幕并执行控制
    result = nn_integration.process_screen_and_control(mock_screenshot)
    print(f"4. 处理结果: {result['success']}")
    if "executed_commands" in result:
        print(f"   执行的命令数: {len(result['executed_commands'])}")
        for cmd in result['executed_commands']:
            print(f"   - {cmd}")
    
    print("\n=== 演示结束 ===")
    print("在实际应用中，请提供真实的模型路径和集成到现有的AndyL AI系统中。")