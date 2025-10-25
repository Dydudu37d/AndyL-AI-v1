#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建示例数据集用于测试AndyL神经网络训练脚本
"""

import os
import numpy as np
import cv2
import random
import argparse
from tqdm import tqdm

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)

def create_sample_image(width=84, height=84, color=(255, 0, 0), shape_type='circle'):
    """创建一个简单的示例图像"""
    # 创建空白图像
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 根据形状类型绘制不同的形状
    if shape_type == 'circle':
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        cv2.circle(image, center, radius, color, -1)
    elif shape_type == 'rectangle':
        pt1 = (width // 4, height // 4)
        pt2 = (width * 3 // 4, height * 3 // 4)
        cv2.rectangle(image, pt1, pt2, color, -1)
    elif shape_type == 'triangle':
        # 修复三角形顶点数组的构造
        pts = np.array([
            [width // 2, height // 4],
            [width // 4, height * 3 // 4],
            [width * 3 // 4, height * 3 // 4]
        ], dtype=np.int32)
        cv2.fillPoly(image, [pts], color)
    elif shape_type == 'ellipse':
        center = (width // 2, height // 2)
        axes = (width // 3, height // 4)
        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)
    
    # 添加一些随机噪声
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    noisy_image = cv2.addWeighted(image, 0.9, noise, 0.1, 0)
    
    return noisy_image

def create_sample_dataset(output_dir='sample_dataset', num_classes=4, images_per_class=50, image_size=84):
    """创建示例数据集"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 定义类别名称和对应的形状
    class_names = ['circle', 'rectangle', 'triangle', 'ellipse']
    class_names = class_names[:num_classes]  # 如果请求的类别数少于4个
    
    # 定义颜色列表
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
    ]
    
    # 为每个类别创建图像
    for class_idx, class_name in enumerate(class_names):
        # 创建类别目录
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        print(f"创建 {class_name} 类别的图像...")
        
        # 为当前类别创建图像
        for i in tqdm(range(images_per_class)):
            # 随机选择颜色
            color = random.choice(colors)
            
            # 创建图像
            image = create_sample_image(
                width=image_size,
                height=image_size,
                color=color,
                shape_type=class_name
            )
            
            # 保存图像
            image_path = os.path.join(class_dir, f"{class_name}_{i:03d}.png")
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCV使用BGR格式
    
    print(f"示例数据集已创建在 {output_dir} 目录中")
    print(f"包含 {num_classes} 个类别，每个类别 {images_per_class} 张图像")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='创建示例数据集')
    parser.add_argument('--output_dir', type=str, default='sample_dataset', help='输出目录')
    parser.add_argument('--num_classes', type=int, default=4, help='类别数量')
    parser.add_argument('--images_per_class', type=int, default=50, help='每个类别的图像数量')
    parser.add_argument('--image_size', type=int, default=84, help='图像大小')
    args = parser.parse_args()
    
    # 创建示例数据集
    create_sample_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        image_size=args.image_size
    )

if __name__ == "__main__":
    main()