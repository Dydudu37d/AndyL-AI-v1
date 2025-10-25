# AndyL神经网络模型加载工具

# 本文件提供加载AndyL神经网络系统中pth或onnx格式模型的功能

import torch
import torch.nn as nn
import onnxruntime
import numpy as np
import os
import logging
import json
from typing import Tuple, Optional, Dict, Any, Union

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AndyLModelLoader")

class AndyLModelLoader:
    """
    AndyL神经网络模型加载工具类
    提供加载PyTorch模型(.pth)和ONNX模型(.onnx)的功能
    """
    
    def __init__(self):
        # 检查必要库是否可用
        self.torch_available = self._check_library_availability('torch')
        self.onnxruntime_available = self._check_library_availability('onnxruntime')
        
        if not self.torch_available:
            logger.warning("PyTorch未安装，无法加载.pth模型")
        if not self.onnxruntime_available:
            logger.warning("ONNX Runtime未安装，无法加载.onnx模型")
    
    def _check_library_availability(self, library_name: str) -> bool:
        """检查指定库是否可用"""
        try:
            __import__(library_name)
            return True
        except ImportError:
            return False
    
    def load_pth_model(self, model_path: str, model_class: Optional[nn.Module] = None) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]]]:
        """
        加载PyTorch模型(.pth)
        
        参数:
            model_path: 模型文件路径
            model_class: 可选的模型类，用于加载自定义模型结构
        
        返回:
            Tuple[Optional[nn.Module], Optional[Dict[str, Any]]]: 加载的模型和元数据
        """
        if not self.torch_available:
            logger.error("PyTorch未安装，无法加载pth模型")
            return None, None
        
        try:
            # 检查文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return None, None
            
            # 加载模型数据
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # 提取元数据
            metadata = checkpoint.get('metadata', {})
            
            # 检查是否只包含状态字典
            if 'model_state_dict' in checkpoint:
                # 需要提供模型类来加载状态字典
                if model_class is None:
                    logger.error("加载包含model_state_dict的模型时需要提供model_class参数")
                    return None, metadata
                
                # 创建模型实例
                model = model_class()
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()  # 设置为评估模式
                logger.info(f"成功加载pth模型: {model_path}")
                return model, metadata
            
            # 处理DQNAgent格式的模型
            elif 'policy_net_state_dict' in checkpoint:
                # 这种情况通常是来自DQNAgent的模型
                if model_class is None:
                    logger.warning("检测到DQNAgent格式的模型，但未提供model_class参数")
                    return None, metadata
                
                model = model_class()
                model.load_state_dict(checkpoint['policy_net_state_dict'])
                model.eval()
                logger.info(f"成功加载DQNAgent格式pth模型: {model_path}")
                return model, metadata
            
            # 假设直接保存了模型
            elif isinstance(checkpoint, nn.Module):
                model = checkpoint
                model.eval()
                logger.info(f"成功加载直接保存的pth模型: {model_path}")
                return model, metadata
            
            else:
                logger.error(f"未知的pth模型格式: {model_path}")
                return None, metadata
            
        except Exception as e:
            logger.error(f"加载pth模型失败: {str(e)}")
            return None, None
    
    def load_onnx_model(self, model_path: str) -> Tuple[Optional[onnxruntime.InferenceSession], Optional[Dict[str, Any]]]:
        """
        加载ONNX模型(.onnx)
        
        参数:
            model_path: ONNX模型文件路径
        
        返回:
            Tuple[Optional[onnxruntime.InferenceSession], Optional[Dict[str, Any]]]: ONNX会话和元数据
        """
        if not self.onnxruntime_available:
            logger.error("ONNX Runtime未安装，无法加载onnx模型")
            return None, None
        
        try:
            # 检查文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return None, None
            
            # 加载ONNX模型
            session = onnxruntime.InferenceSession(model_path)
            logger.info(f"成功加载onnx模型: {model_path}")
            
            # ONNX模型本身不包含元数据，我们尝试从同名.json文件加载
            metadata_path = os.path.splitext(model_path)[0] + '.json'
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    logger.info(f"成功加载模型元数据: {metadata_path}")
                except Exception as e:
                    logger.warning(f"加载元数据失败: {str(e)}")
            
            return session, metadata
            
        except Exception as e:
            logger.error(f"加载onnx模型失败: {str(e)}")
            return None, None
    
    def load_model(self, model_path: str, model_class: Optional[nn.Module] = None) -> Tuple[Optional[Union[nn.Module, onnxruntime.InferenceSession]], Optional[Dict[str, Any]], str]:
        """
        自动检测并加载模型
        
        参数:
            model_path: 模型文件路径
            model_class: 可选的模型类，用于加载自定义pth模型结构
        
        返回:
            Tuple[模型实例或ONNX会话, 元数据, 模型类型]
        """
        # 获取文件扩展名
        _, ext = os.path.splitext(model_path.lower())
        
        if ext == '.pth':
            model, metadata = self.load_pth_model(model_path, model_class)
            return model, metadata, 'pth'
        elif ext == '.onnx':
            session, metadata = self.load_onnx_model(model_path)
            return session, metadata, 'onnx'
        else:
            logger.error(f"不支持的模型格式: {ext}")
            return None, None, 'unknown'

# 提供一个简单的函数接口
def load_andyL_model(model_path: str, model_class: Optional[nn.Module] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]], str]:
    """
    加载AndyL神经网络模型的便捷函数
    
    参数:
        model_path: 模型文件路径
        model_class: 可选的模型类，用于加载自定义pth模型结构
    
    返回:
        Tuple[模型实例或ONNX会话, 元数据, 模型类型]
    """
    loader = AndyLModelLoader()
    return loader.load_model(model_path, model_class)

# 示例用法
if __name__ == "__main__":
    # 示例：创建一个简单的CNN模型类用于测试加载
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 21 * 21, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 21 * 21)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 尝试加载示例模型
    models_dir = "saved_models"
    
    # 尝试加载pth模型
    pth_model_path = os.path.join(models_dir, "andyL_model.pth")
    print(f"\n尝试加载pth模型: {pth_model_path}")
    pth_model, pth_metadata, pth_type = load_andyL_model(pth_model_path, SimpleCNN)
    if pth_model:
        print(f"成功加载{pth_type}模型")
        print(f"元数据: {pth_metadata}")
    
    # 尝试加载onnx模型
    onnx_model_path = os.path.join(models_dir, "andyL_model.onnx")
    print(f"\n尝试加载onnx模型: {onnx_model_path}")
    onnx_session, onnx_metadata, onnx_type = load_andyL_model(onnx_model_path)
    if onnx_session:
        print(f"成功加载{onnx_type}模型")
        print(f"元数据: {onnx_metadata}")
        print(f"输入名称: {[input.name for input in onnx_session.get_inputs()]}")
        print(f"输出名称: {[output.name for output in onnx_session.get_outputs()]}")
    
    print("\n模型加载工具使用指南：")
    print("1. 加载pth模型: model, metadata, type = load_andyL_model('path/to/model.pth', ModelClass)")
    print("2. 加载onnx模型: session, metadata, type = load_andyL_model('path/to/model.onnx')")
    print("3. 确保已安装必要依赖: pip install torch onnxruntime")