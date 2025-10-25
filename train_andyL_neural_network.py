#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AndyL神经网络训练脚本
此脚本用于训练神经网络模型，生成可在AndyL AI系统中使用的.pth或.onnx模型文件

使用方法:
  python train_andyL_neural_network.py --data_dir path/to/data --model_type cnn --epochs 50
"""

import os
import sys
import logging
import argparse
import time
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 强制设置Keras后端为tensorflow，避免mxnet导入错误
os.environ['KERAS_BACKEND'] = 'tensorflow'

# 配置日志
# 首先创建一个logger
logger = logging.getLogger("AndyLNNTrainer")
logger.setLevel(logging.INFO)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建文件处理器并设置编码为UTF-8
file_handler = logging.FileHandler("training.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 创建控制台处理器并设置编码为UTF-8（针对Windows系统）
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# 将处理器添加到logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class ImageDataset(data.Dataset):
    """\图像数据集类，用于加载和预处理训练数据"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            # 读取图像
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"无法读取图像: {image_path}")
            
            # 转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # 获取标签
            label = self.labels[idx]
            
            return image, label
        except Exception as e:
            logger.error(f"处理图像时出错 ({self.image_paths[idx]}): {e}")
            # 返回一个零张量和默认标签
            return torch.zeros((3, 84, 84)), 0

class SimpleCNN(nn.Module):
    """简单的CNN模型，适用于图像分类任务"""
    
    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (3, 84, 84)):
        super(SimpleCNN, self).__init__()
        
        # 计算输入通道数
        in_channels = input_shape[0]
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 计算全连接层输入大小
        # 经过两次池化后，尺寸变为 (input_size / 2^2)
        h, w = input_shape[1] // 4, input_shape[2] // 4
        fc_input_size = 64 * h * w
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(0.5)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 展平
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class AndyLNeuralNetworkTrainer:
    """AndyL神经网络训练器类"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 检查并创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 最佳模型指标
        self.best_val_accuracy = 0.0
        
        # 用于记录训练历史
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
    
    def _create_model(self) -> nn.Module:
        """创建神经网络模型"""
        model_type = self.config['model_type'].lower()
        
        if model_type == 'cnn':
            model = SimpleCNN(
                num_classes=self.config['num_classes'],
                input_shape=(self.config['channels'], self.config['image_height'], self.config['image_width'])
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 移至设备
        model = model.to(self.device)
        logger.info(f"创建模型: {model_type}")
        logger.debug(f"模型结构: {model}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config['optimizer'].lower()
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return optimizer
    
    def prepare_data(self) -> None:
        """准备训练、验证和测试数据"""
        data_dir = self.config['data_dir']
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise Exception(f"数据目录不存在: {data_dir}")
        
        # 假设数据按照类别组织在子目录中
        image_paths = []
        labels = []
        class_names = sorted(os.listdir(data_dir))
        
        # 确保类别数量正确
        if len(class_names) != self.config['num_classes']:
            logger.warning(f"类别数量不匹配: 期望 {self.config['num_classes']}, 实际 {len(class_names)}")
        
        # 收集图像路径和标签
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # 获取该类别下的所有图像文件
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_file)
                    image_paths.append(img_path)
                    labels.append(class_idx)
        
        if len(image_paths) == 0:
            raise Exception(f"数据目录中没有找到图像文件: {data_dir}")
        
        logger.info(f"找到 {len(image_paths)} 个图像文件，共 {len(class_names)} 个类别")
        
        # 分割数据集
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=self.config['test_size'], random_state=self.config['random_seed']
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=self.config['val_size'], random_state=self.config['random_seed']
        )
        
        logger.info(f"训练集: {len(train_paths)} 样本")
        logger.info(f"验证集: {len(val_paths)} 样本")
        logger.info(f"测试集: {len(test_paths)} 样本")
        
        # 定义数据变换
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image_height'], self.config['image_width'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image_height'], self.config['image_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = ImageDataset(val_paths, val_labels, transform=val_test_transform)
        test_dataset = ImageDataset(test_paths, test_labels, transform=val_test_transform)
        
        # 创建数据加载器
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers']
        )
        
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers']
        )
        
        self.test_loader = data.DataLoader(
            test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers']
        )
        
        # 保存类别名称
        self.class_names = class_names
        with open(os.path.join(self.config['output_dir'], 'class_names.txt'), 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # 移至设备
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            running_loss += loss.item()
            
            # 收集预测结果
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 打印进度
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                logger.info(f"Epoch {epoch+1}/{self.config['epochs']}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算平均损失和准确率
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        
        # 记录到训练历史
        self.train_history['loss'].append(epoch_loss)
        self.train_history['accuracy'].append(epoch_accuracy)
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """在验证集上评估模型"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                # 移至设备
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 记录损失
                running_loss += loss.item()
                
                # 收集预测结果
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和准确率
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        
        # 记录到验证历史
        self.val_history['loss'].append(val_loss)
        self.val_history['accuracy'].append(val_accuracy)
        
        # 如果验证准确率是目前最好的，保存模型
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.save_model(f"best_model.pth")
            logger.info(f"新的最佳模型已保存，验证准确率: {val_accuracy:.4f}")
        
        return val_loss, val_accuracy
    
    def test(self) -> Dict:
        """在测试集上评估模型"""
        # 加载最佳模型
        best_model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                # 移至设备
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 收集预测结果
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算评估指标
        test_accuracy = accuracy_score(all_labels, all_predictions)
        
        # 生成详细的分类报告
        class_report = classification_report(
            all_labels, all_predictions, target_names=self.class_names
        )
        
        logger.info(f"测试准确率: {test_accuracy:.4f}")
        logger.info(f"分类报告:\n{class_report}")
        
        # 保存测试结果，使用UTF-8编码
        with open(os.path.join(self.config['output_dir'], 'test_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"测试准确率: {test_accuracy:.4f}\n\n")
            f.write(f"分类报告:\n{class_report}")
        
        return {
            'accuracy': test_accuracy,
            'classification_report': class_report
        }
    
    def train(self) -> None:
        """执行完整的训练流程"""
        logger.info("开始训练...")
        start_time = time.time()
        
        # 准备数据
        self.prepare_data()
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            logger.info(f"===== Epoch {epoch+1}/{self.config['epochs']} =====")
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            
            # 在验证集上评估
            val_loss, val_acc = self.validate(epoch)
            logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 学习率调度（如果启用）
            if self.config.get('use_lr_scheduler', False) and hasattr(self, 'scheduler'):
                self.scheduler.step(val_loss)
            
            # 每个epoch保存一次模型（可选）
            if self.config.get('save_every_epoch', False):
                self.save_model(f"model_epoch_{epoch+1}.pth")
        
        # 在测试集上评估最终模型
        logger.info("===== 测试最佳模型 =====")
        self.test()
        
        # 导出为ONNX格式（可选）
        if self.config.get('export_onnx', False):
            self.export_to_onnx()
        
        # 保存训练历史到JSON文件
        self.save_training_history()
        
        total_time = time.time() - start_time
        logger.info(f"训练完成！总耗时: {total_time:.2f}秒")
        logger.info(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
    
    def save_training_history(self) -> None:
        """保存训练历史到文件"""
        history_path = os.path.join(self.config['output_dir'], 'training_history.txt')
        with open(history_path, 'w', encoding='utf-8') as f:
            f.write("===== 训练历史 =====\n\n")
            
            # 保存训练损失和准确率
            f.write("训练指标:\n")
            f.write("Epoch\tLoss\tAccuracy\n")
            for i, (loss, acc) in enumerate(zip(self.train_history['loss'], self.train_history['accuracy'])):
                f.write(f"{i+1}\t{loss:.6f}\t{acc:.6f}\n")
            
            f.write("\n验证指标:\n")
            f.write("Epoch\tLoss\tAccuracy\n")
            for i, (loss, acc) in enumerate(zip(self.val_history['loss'], self.val_history['accuracy'])):
                f.write(f"{i+1}\t{loss:.6f}\t{acc:.6f}\n")
        
        logger.info(f"训练历史已保存到: {history_path}")
    
    def save_model(self, filename: str) -> None:
        """保存模型到文件"""
        model_path = os.path.join(self.config['output_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_accuracy': self.best_val_accuracy
        }, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """从文件加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_val_accuracy' in checkpoint:
            self.best_val_accuracy = checkpoint['best_val_accuracy']
        logger.info(f"模型已从: {model_path} 加载")
    
    def export_to_onnx(self) -> None:
        """将模型导出为ONNX格式"""
        try:
            # 创建一个示例输入
            dummy_input = torch.randn(1, self.config['channels'], 
                                      self.config['image_height'], 
                                      self.config['image_width'], 
                                      device=self.device)
            
            # 导出模型
            onnx_path = os.path.join(self.config['output_dir'], 'andyL_model.onnx')
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info(f"模型已导出为ONNX格式: {onnx_path}")
        except Exception as e:
            logger.error(f"导出ONNX模型失败: {e}")

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AndyL神经网络训练脚本')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./models', help='模型输出目录')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数量')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例（相对于训练集）')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn'], help='模型类型')
    parser.add_argument('--image_height', type=int, default=84, help='输入图像高度')
    parser.add_argument('--image_width', type=int, default=84, help='输入图像宽度')
    parser.add_argument('--channels', type=int, default=3, help='输入图像通道数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减（L2正则化）')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    # 硬件和性能参数
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU进行训练')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的线程数')
    
    # 其他参数
    parser.add_argument('--log_interval', type=int, default=10, help='每多少个批次打印一次日志')
    parser.add_argument('--save_every_epoch', action='store_true', help='每个epoch都保存模型')
    parser.add_argument('--export_onnx', action='store_true', help='导出为ONNX格式')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 如果启用详细日志，设置日志级别为DEBUG
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 创建配置字典
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'num_classes': args.num_classes,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'model_type': args.model_type,
        'image_height': args.image_height,
        'image_width': args.image_width,
        'channels': args.channels,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'random_seed': args.random_seed,
        'use_gpu': args.use_gpu,
        'num_workers': args.num_workers,
        'log_interval': args.log_interval,
        'save_every_epoch': args.save_every_epoch,
        'export_onnx': args.export_onnx
    }
    
    # 打印配置信息
    logger.info("===== 训练配置 =====")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    # 创建训练器并开始训练
    trainer = AndyLNeuralNetworkTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()