# AndyL神经网络模型保存示例

# 本文件提供如何在AndyL神经网络系统中保存pth或onnx格式模型的示例代码

import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AndyLModelSaver")

class AndyLModelSaver:
    """
    AndyL神经网络模型保存工具类
    提供保存PyTorch模型(.pth)和导出ONNX模型(.onnx)的功能
    """
    
    def __init__(self):
        # 确保PyTorch可用
        self.torch_available = self._check_torch_availability()
        if not self.torch_available:
            logger.warning("PyTorch未安装，无法使用模型保存功能")
    
    def _check_torch_availability(self):
        """检查PyTorch是否可用"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def save_pth_model(self, model, save_path, metadata=None):
        """
        保存PyTorch模型为.pth格式
        
        参数:
            model: PyTorch模型实例
            save_path: 保存路径
            metadata: 可选的元数据字典
        
        返回:
            bool: 保存是否成功
        """
        if not self.torch_available:
            logger.error("PyTorch未安装，无法保存pth模型")
            return False
        
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 准备保存的数据
            save_data = {
                'model_state_dict': model.state_dict()
            }
            
            # 添加元数据
            if metadata:
                save_data['metadata'] = metadata
            
            # 保存模型
            torch.save(save_data, save_path)
            logger.info(f"模型已成功保存为pth格式: {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存pth模型失败: {str(e)}")
            return False
    
    def export_onnx_model(self, model, input_shape, onnx_path, dynamic_axes=None):
        """
        将PyTorch模型导出为ONNX格式
        
        参数:
            model: PyTorch模型实例
            input_shape: 输入张量形状 (batch_size, channels, height, width)
            onnx_path: ONNX文件保存路径
            dynamic_axes: 动态维度配置，用于支持可变输入尺寸
        
        返回:
            bool: 导出是否成功
        """
        if not self.torch_available:
            logger.error("PyTorch未安装，无法导出ONNX模型")
            return False
        
        try:
            # 设置为评估模式
            model.eval()
            
            # 创建示例输入张量
            dummy_input = torch.randn(*input_shape)
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            
            # 导出ONNX模型
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes or {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"模型已成功导出为ONNX格式: {onnx_path}")
            return True
        except Exception as e:
            logger.error(f"导出ONNX模型失败: {str(e)}")
            return False

# 示例: 如何扩展AndyLNeuralNetwork类以支持模型保存
class AndyLNeuralNetworkWithSaving:
    """
    扩展的AndyLNeuralNetwork类，增加模型保存功能
    注意：这是示例实现，实际使用时需要与原有AndyLNeuralNetwork类兼容
    """
    
    def __init__(self):
        # 原始AndyLNeuralNetwork初始化逻辑
        self.model = None  # 在实际使用中，这里应该是真实的PyTorch模型
        self.model_path = None
        self.model_type = "pytorch"  # 假设使用PyTorch
        self.input_shape = (84, 84, 3)  # 默认输入形状
        
        # 初始化模型保存工具
        self.model_saver = AndyLModelSaver()
    
    def save_model(self, save_path, model_format="pth", metadata=None):
        """
        保存或导出模型
        
        参数:
            save_path: 保存路径
            model_format: 保存格式，可选 "pth" 或 "onnx"
            metadata: 可选的元数据
        
        返回:
            bool: 保存是否成功
        """
        if self.model is None:
            logger.error("模型未加载，无法保存")
            return False
        
        if model_format.lower() == "pth":
            # 确保文件扩展名为.pth
            if not save_path.endswith('.pth'):
                save_path += '.pth'
            return self.model_saver.save_pth_model(self.model, save_path, metadata)
        elif model_format.lower() == "onnx":
            # 确保文件扩展名为.onnx
            if not save_path.endswith('.onnx'):
                save_path += '.onnx'
            
            # 准备输入形状 (batch_size=1, channels=3, height, width)
            input_shape = (1, 3, self.input_shape[0], self.input_shape[1])
            return self.model_saver.export_onnx_model(self.model, input_shape, save_path)
        else:
            logger.error(f"不支持的模型格式: {model_format}")
            return False

# 示例用法
if __name__ == "__main__":
    # 假设我们有一个简单的PyTorch模型
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
    
    # 创建示例模型
    example_model = SimpleCNN()
    
    # 创建模型保存工具
    saver = AndyLModelSaver()
    
    # 定义保存路径
    models_dir = "saved_models"
    pth_path = os.path.join(models_dir, "andyL_model.pth")
    onnx_path = os.path.join(models_dir, "andyL_model.onnx")
    
    # 保存为pth格式
    metadata = {
        'model_name': 'AndyLExampleModel',
        'version': '1.0',
        'input_shape': (3, 84, 84),
        'description': '示例CNN模型'
    }
    saver.save_pth_model(example_model, pth_path, metadata)
    
    # 导出为ONNX格式
    input_shape = (1, 3, 84, 84)  # (batch_size, channels, height, width)
    saver.export_onnx_model(example_model, input_shape, onnx_path)
    
    print("\n=== 模型保存示例完成 ===")
    print(f"PTH模型已保存到: {pth_path}")
    print(f"ONNX模型已导出到: {onnx_path}")
    print("\n在实际应用中，请将此功能集成到AndyL神经网络系统中。")