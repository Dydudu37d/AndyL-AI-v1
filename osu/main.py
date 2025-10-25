import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import ImageGrab, Image, ImageOps
import cv2
import time
import sys
import pyautogui
import traceback

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
            return predicted

# Hyperparameters
batch_size = 128
learning_rate = 0.005
num_epochs = 100

# Load MNIST dataset and apply transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.exists(os.path.join(MODEL_DIR, "model.pth")):
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pth")))

# Training loop
if "train" in sys.argv:
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move data to the same device as the model
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))

# 评估模型在测试集上的性能
test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# 测试模式
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    # 从测试集中获取一些样本用于预测和可视化
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    
    # 显示一些预测结果
    print("\n===== 模型预测演示 =====")
    # 使用当前批次的第一张图像进行预测演示
    sample_image = test_images[:1]
    sample_label = test_labels[:1]
    prediction = model.predict(sample_image)
    print(f"实际标签: {sample_label.item()}")
    print(f"预测结果: {prediction.item()}")
    
    # 完整测试集评估
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\n测试集准确率: {accuracy:.2f}%")

print("\n训练和评估完成！")

# 对缩小10倍的屏幕截图进行目标检测，并返回放大10倍后的目标坐标
def recognize_scaled_digits(model, transform, device, test_image_path=None, scale_factor=10):
    """从缩小的屏幕截图识别数字并返回原始比例的坐标
    
    Args:
        model: 训练好的数字识别模型
        transform: 图像变换函数
        device: 运行设备
        test_image_path: 可选的测试图像路径
        scale_factor: 缩放因子（默认10倍）
    
    Returns:
        dict: 包含识别结果和放大后的坐标信息
    """
    # 确保目录存在
    os.makedirs('osu', exist_ok=True)
    
    # 创建一个简单的日志文件，在函数开始就写入一条信息
    with open(os.path.join(ROOT_DIR, 'scaled_startup_log.txt'), 'w') as f:
        f.write("缩放数字识别程序开始执行\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"缩放因子: {scale_factor}x\n")
        if test_image_path:
            f.write(f"测试图像路径: {test_image_path}\n")
    
    # 创建日志文件
    log_path = os.path.join(ROOT_DIR, 'scaled_execution_log.txt')
    log_file = open(log_path, 'w')
    log_file.write("===== 缩放数字识别 =====\n")
    log_file.write(f"日志文件创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"缩放因子: {scale_factor}x\n")
    log_file.flush()
    
    def log_message(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_message("\n===== 缩放数字识别 =====")
    log_message(f"日志功能已初始化，文件路径: {log_path}")
    log_message(f"使用缩放因子: {scale_factor}x")
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"使用测试图像: {test_image_path}")
        screenshot_gray = Image.open(test_image_path).convert('L')
    else:
        
        # 截取屏幕
        screenshot = ImageGrab.grab()
        screenshot_gray = screenshot.convert('L')
    
    # 保存原始截图以便调试
    screenshot_gray.save(os.path.join(ROOT_DIR, 'original_screenshot.png'))
    print("原始屏幕截图已保存到 original_screenshot.png")
    
    # 获取原始图像尺寸
    original_width, original_height = screenshot_gray.size
    log_message(f"原始图像尺寸: {original_width}x{original_height}")
    
    # 缩小图像scale_factor倍
    small_width = original_width // scale_factor
    small_height = original_height // scale_factor
    log_message(f"缩小后的图像尺寸: {small_width}x{small_height}")
    
    # 调整图像大小（缩小）
    small_img = screenshot_gray.resize((small_width, small_height), Image.LANCZOS)
    small_img.save(os.path.join(ROOT_DIR, 'scaled_down_screenshot.png'))
    log_message("缩小后的图像已保存到 scaled_down_screenshot.png")
    
    # 转换为OpenCV格式进行图像处理
    img_cv = np.array(small_img)
    
    # 自动检测文字颜色（白字黑底或黑字白底）
    # 计算图像的平均亮度
    avg_brightness = np.mean(img_cv)
    # 如果平均亮度较低，很可能是黑底
    is_dark_background = avg_brightness < 128
    print(f"图像平均亮度: {avg_brightness:.2f}, 检测为{'黑底白字' if is_dark_background else '白底黑字'}")
    
    # 根据背景类型选择处理方式
    if is_dark_background:
        # 对于黑底白字，我们可以直接处理
        print("检测到黑底白字，应用相应处理...")
        # 对黑底白字图像进行反转，使其变成白底黑字以便处理
        img_cv_inverted = 255 - img_cv
        cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_inverted.png'), img_cv_inverted)
        print("反转后的图像已保存到 scaled_inverted.png")
        img_processed = img_cv_inverted
    else:
        # 对于白底黑字，正常处理
        img_processed = img_cv
    
    # 增加对比度（提高数字和背景的区分度）
    # 为不同背景类型使用不同的对比度参数
    if is_dark_background:
        # 为黑底白字图像应用更强的对比度增强
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    else:
        # 为白底黑字图像应用常规对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_processed)
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_contrast_enhanced.png'), img_clahe)
    print("对比度增强后的图像已保存到 scaled_contrast_enhanced.png")
    
    # 应用高斯模糊减少噪点
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)  # 缩小后的图像使用更小的模糊核
    
    # 添加中值滤波进一步减少噪点
    img_median = cv2.medianBlur(img_blur, 3)  # 缩小后的图像使用更小的滤波核
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_blurred.png'), img_median)
    
    # 尝试多种阈值方法
    # 为不同背景类型使用不同的阈值处理策略
    if is_dark_background:
        print("为黑底白字图像应用特殊阈值处理...")
        # 黑底白字（已反转）情况下，尝试多种不同的阈值
        _, thresh1 = cv2.threshold(img_median, 90, 255, cv2.THRESH_BINARY_INV)  # 更低的阈值
        _, thresh1_alt = cv2.threshold(img_median, 70, 255, cv2.THRESH_BINARY_INV)  # 尝试极低的阈值
        
        # 尝试多种自适应阈值参数，使用更小的块大小和更强的常数
        thresh2 = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 5, 6)  # 更小的块大小，更强的常数
        thresh3 = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 5, 6)
        
        # 保存额外的阈值图像用于调试
        cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_threshold_low.png'), thresh1_alt)
        
        # 尝试合并更多阈值结果
        combined_thresh = cv2.bitwise_or(thresh1, thresh1_alt)
        combined_thresh = cv2.bitwise_or(combined_thresh, thresh2)
        combined_thresh = cv2.bitwise_or(combined_thresh, thresh3)
    else:
        print("为白底黑字图像应用特殊阈值处理...")
        # 调整为较低的阈值以捕获更暗的数字
        _, thresh1 = cv2.threshold(img_median, 150, 255, cv2.THRESH_BINARY_INV)  # 增加阈值以减少背景干扰
        
        # 尝试多种自适应阈值参数
        thresh2 = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 3)  # 增加常数以减少噪声
        thresh3 = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 5)
        
        # 合并三种阈值结果以提高检测率
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
        combined_thresh = cv2.bitwise_or(combined_thresh, thresh3)
    
    # 应用形态学操作来增强轮廓
    # 为不同背景类型使用不同的形态学操作
    if is_dark_background:
        # 黑底白字情况下使用更轻柔的形态学操作，避免过度处理
        print("为黑底白字图像应用特殊形态学操作...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # 尝试轻微的开运算来去除噪声
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
    else:
        # 对于白底黑字，使用更精细的形态学操作来去除大的背景轮廓
        print("为白底黑字图像应用精细形态学操作...")
        # 先进行腐蚀操作，尝试断开大轮廓
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 缩小后的图像使用更小的核
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_ERODE, kernel_erode, iterations=1)
        
        # 然后进行小的膨胀操作以保持数字形状
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_DILATE, kernel_dilate, iterations=1)
    
    # 保存所有阈值图像用于调试
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_threshold_simple.png'), thresh1)
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_threshold_adaptive1.png'), thresh2)
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_threshold_adaptive2.png'), thresh3)
    
    # 保存阈值图像用于调试
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_threshold.png'), combined_thresh)
    print("阈值处理后的图像已保存到 scaled_threshold.png")
    
    # 查找轮廓 - 使用RETR_LIST获取所有轮廓，而不仅仅是外部轮廓
    contours, _ = cv2.findContours(combined_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"找到 {len(contours)} 个轮廓")
    
    recognized_digits = []
    
    # 创建用于标记的彩色图像
    if is_dark_background:
        # 对于黑底白字，使用反转后的图像作为基础
        marked_img = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGB)
    else:
        # 对于白底黑字，使用处理后的图像
        marked_img = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGB)
    
    # 为缩小的图像调整过滤条件
    log_message("为缩小的图像应用调整后的轮廓过滤条件...")
    # 缩小后的图像使用更小的过滤参数
    min_area = 3  # 缩小的最小面积，降低以捕获更多小数字
    max_area = 400  # 缩小的最大面积，增加以捕获更多数字
    min_width = 1  # 缩小的最小宽度，降低以捕获数字1
    min_height = 2  # 缩小的最小高度，降低以捕获更多小数字
    max_width = 25  # 缩小的最大宽度，增加以捕获更大数字
    max_height = 40  # 缩小的最大高度，增加以捕获更大数字
    
    log_message(f"使用过滤条件: min_area={min_area}, max_area={max_area}, min_width={min_width}, min_height={min_height}, max_width={max_width}, max_height={max_height}")
    
    # 添加轮廓过滤计数器
    filtered_contours_count = 0
    skipped_contours_count = 0
    
    log_message(f"开始过滤 {len(contours)} 个轮廓...")
    
    for i, contour in enumerate(contours):
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 计算宽高比
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # 检测可能的数字1轮廓（宽高比非常小的垂直轮廓）
        is_possible_one = aspect_ratio < 0.2 and w > 3 and h > 8 and area > 8 and area < 50
        
        # 添加详细的轮廓信息（每100个轮廓记录一次，避免日志过大）
        if i % 100 == 0:
            log_message(f"处理轮廓 {i}: 位置=({x},{y}), 大小={w}x{h}, 面积={area:.2f}, 宽高比={aspect_ratio:.2f}")
        
        # 对于可能是数字1的轮廓，应用更宽松的过滤条件
        if is_possible_one:
            # 跳过标准过滤条件，使用特殊的数字1过滤条件
            # 只做最基本的大小检查
            if w < 2 or h < 5 or area < 5 or area > 80:
                skipped_contours_count += 1
                continue
        else:
            # 统一的过滤条件，适用于所有其他轮廓
            # 1. 面积过滤
            if area < min_area or (not is_dark_background and area > max_area):
                skipped_contours_count += 1
                continue
            
            # 2. 尺寸过滤
            if w < min_width or h < min_height or (not is_dark_background and (w > max_width or h > max_height)):
                skipped_contours_count += 1
                continue
        
        # 3. 宽高比过滤 - 进一步放宽以确保数字1能通过
        if (aspect_ratio < 0.01 or aspect_ratio > 2.0):
            skipped_contours_count += 1
            continue
        
        # 添加调试日志，记录所有宽高比小于0.3的轮廓，可能包含数字1
        if aspect_ratio < 0.3:
            log_message(f"  低宽高比轮廓: ID={i}, 宽高比={aspect_ratio:.3f}, 面积={area:.2f}, 尺寸={w}x{h}")
        
        # 4. 额外的形状验证 - 检查轮廓的紧凑度
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            # 降低紧凑度下限以允许数字1这样的细长形状通过
            if compactness < 0.01 or compactness > 0.8:
                skipped_contours_count += 1
                continue
        
        # 通过所有过滤条件
        filtered_contours_count += 1
        if filtered_contours_count <= 50:
            log_message(f"  ✓ 保留轮廓 {i}: 面积={area:.2f}, 尺寸={w}x{h}, 宽高比={aspect_ratio:.2f}")
        
        # 扩展边界框以确保包含整个数字
        margin = max(1, int(min(w, h) * 0.1))  # 缩小的margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_cv.shape[1] - x, w + 2 * margin)
        h = min(img_cv.shape[0] - y, h + 2 * margin)
        
        # 提取数字区域
        digit_roi = combined_thresh[y:y+h, x:x+w]
        
        # 调整大小为28x28并保持长宽比
        digit_pil = Image.fromarray(digit_roi)
        
        # 添加形态学操作以更好地处理数字形状，特别是细长的数字1
        if aspect_ratio < 0.2:  # 可能是数字1，收紧条件
            # 转换为OpenCV格式进行形态学操作
            roi_cv = np.array(digit_pil)
            # 应用膨胀操作使线条更粗，更适合数字1的特征
            kernel = np.ones((2, 2), np.uint8)  # 缩小后的图像使用更小的核
            roi_cv = cv2.dilate(roi_cv, kernel, iterations=1)  # 减少迭代次数
            digit_pil = Image.fromarray(roi_cv)
        # 对其他数字进行常规形态学处理
        else:
            roi_cv = np.array(digit_pil)
            kernel = np.ones((1, 1), np.uint8)
            roi_cv = cv2.morphologyEx(roi_cv, cv2.MORPH_CLOSE, kernel)  # 闭运算填充小缺口
            digit_pil = Image.fromarray(roi_cv)
        
        # 添加白色边框以保持长宽比
        size = max(w, h)
        square_img = Image.new('L', (size, size), 0)
        paste_x = (size - w) // 2
        paste_y = (size - h) // 2
        square_img.paste(digit_pil, (paste_x, paste_y))
        
        # 调整大小为28x28
        digit_resized = square_img.resize((28, 28), Image.LANCZOS)
        
        # 显示调整后的图像统计信息（每10个数字记录一次）
        digit_array = np.array(digit_resized)
        white_pixels = np.sum(digit_array > 0)
        
        if len(recognized_digits) % 10 == 0:
            log_message(f"  处理数字区域: 白色像素数={white_pixels}/{28*28}")
        
        # 提高白色像素的阈值要求，减少空白或噪点区域的误识别
        min_white_pixels = 12  # 提高阈值以减少误识别
        
        # 如果白色像素太少，可能不是有效的数字，跳过
        if white_pixels < min_white_pixels:
            skipped_contours_count += 1
            continue
        
        # 应用变换
        digit_tensor = transform(digit_resized).unsqueeze(0).to(device)
        
        # 获取置信度和预测结果
        with torch.no_grad():
            outputs = model(digit_tensor)
            
            # 对可能是数字1的轮廓进行特殊处理
            # 收紧条件以减少误识别
            if aspect_ratio < 0.2 and 8 < area < 70 and w > 3 and h > 10:
                log_message(f"  应用数字1特殊处理: 宽高比={aspect_ratio:.3f}, 面积={area:.2f}")
                
                # 调整数字1的预测分数权重
                outputs[0, 1] += 10.0  # 大幅减少偏好增加量
                # 适度降低混淆数字的分数
                outputs[0, 3] -= 15.0  # 增加数字3的惩罚
                outputs[0, 7] -= 10.0  # 增加数字7的惩罚
            
            # 重新计算置信度和预测
            confidence, prediction = torch.max(outputs.data, 1)
            digit = prediction.item()
        
        # 设置置信度阈值
        confidence_threshold = 0.5  # 提高阈值以减少误识别
        
        if confidence.item() >= confidence_threshold:
            # 计算原始比例的坐标（放大scale_factor倍）
            original_x = x * scale_factor
            original_y = y * scale_factor
            original_w = w * scale_factor
            original_h = h * scale_factor
            
            # 存储识别结果，包括原始比例的坐标
            recognized_digits.append((digit, x, y, w, h, original_x, original_y, original_w, original_h, area, confidence.item()))
            if len(recognized_digits) <= 50:
                log_message(f"  ✓ 识别成功: 数字={digit}, 置信度={confidence.item():.4f}, "
                          f"缩放位置=({x},{y}), 原始位置=({original_x},{original_y}), "
                          f"缩放尺寸={w}x{h}, 原始尺寸={original_w}x{original_h}")
        else:
            skipped_contours_count += 1
            if skipped_contours_count % 100 == 0:
                log_message(f"  ✗ 置信度不足: 预测={digit}, 置信度={confidence.item():.4f} < {confidence_threshold}")
        
        # 在图像上标记识别结果
        cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 使用更细的边框
        cv2.putText(marked_img, str(digit), (x, max(0, y - 2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # 更小的字体
        cv2.putText(marked_img, f"ID:{i}", (x, y + h + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # 更小的字体
    
    # 输出过滤统计信息
    log_message(f"\n===== 过滤统计 =====")
    log_message(f"原始轮廓总数: {len(contours)}")
    log_message(f"通过过滤条件的轮廓数: {filtered_contours_count}")
    log_message(f"跳过的轮廓数: {skipped_contours_count}")
    log_message(f"最终识别到的数字数: {len(recognized_digits)}")
    
    # 保存标记后的图像
    cv2.imwrite(os.path.join(ROOT_DIR, 'scaled_recognized_digits.png'), marked_img)
    log_message("标记后的缩放图像已保存到 scaled_recognized_digits.png")
    
    # 创建原始尺寸的标记图像，显示原始比例的坐标
    original_marked_img = cv2.cvtColor(np.array(screenshot_gray), cv2.COLOR_GRAY2RGB)
    
    # 显示识别结果
    results = {
        'recognized_digits': [],
        'min_value': None,
        'min_digits': [],
        'scale_factor': scale_factor,
        'original_image_size': (original_width, original_height),
        'scaled_image_size': (small_width, small_height)
    }
    
    if recognized_digits:
        # 按位置排序数字以保持从左到右的顺序
        recognized_digits.sort(key=lambda x: x[5])  # 按原始x坐标排序
        
        # 去重处理 - 合并位置接近且值相同的数字，避免重复识别
        unique_digits = []
        digit_positions = []
        for digit, x, y, w, h, orig_x, orig_y, orig_w, orig_h, area, confidence in recognized_digits:
            # 检查是否与已添加的数字位置过近
            is_duplicate = False
            for pos_x, pos_y in digit_positions:
                # 如果距离小于数字平均大小的一半，视为重复
                if abs(orig_x - pos_x) < orig_w/2 and abs(orig_y - pos_y) < orig_h/2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_digits.append((digit, orig_x, orig_y, orig_w, orig_h, confidence))
                digit_positions.append((orig_x, orig_y))
                
                # 在原始尺寸图像上标记
                cv2.rectangle(original_marked_img, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 0, 255), 2)
                cv2.putText(original_marked_img, str(digit), (orig_x, max(0, orig_y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        log_message(f"去重后的数字数: {len(unique_digits)}")
        log_message(f"识别到的数字: {[d[0] for d in unique_digits]}")
        
        # 统计每个数字出现的次数
        digit_counts = {}
        for digit, *_ in unique_digits:
            if digit in digit_counts:
                digit_counts[digit] += 1
            else:
                digit_counts[digit] = 1
        log_message(f"数字出现频率: {digit_counts}")
        
        # 找出最小数字，忽略小于1的数字
        filtered_digits = [d[0] for d in unique_digits if d[0] >= 1]
        if filtered_digits:
            min_value = min(filtered_digits)
            min_digits = [d for d in unique_digits if d[0] == min_value]
        else:
            # 如果没有大于等于1的数字，设置默认值和空列表
            log_message("警告: 没有找到大于等于1的数字")
            min_value = None
            min_digits = []
        
        if min_value is not None:
            log_message(f"最小的数字是: {min_value}")
            log_message(f"找到 {len(min_digits)} 个最小数字实例")
            
            # 在原始尺寸图像上高亮显示最小数字
            for digit, x, y, w, h, confidence in min_digits:
                # 绘制更醒目的边框
                cv2.rectangle(original_marked_img, (x-2, y-2), (x + w + 2, y + h + 2), (0, 255, 0), 3)
                # 添加文字说明
                text = f"最小: {digit}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = x
                text_y = y - 10 if y > 20 else y + h + text_size[1] + 10
                # 绘制文字背景
                cv2.rectangle(original_marked_img, 
                            (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5), 
                            (0, 255, 0), -1)
                # 绘制文字
                cv2.putText(original_marked_img, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            log_message("未找到有效的最小数字，跳过高亮显示")
        
        # 保存原始尺寸的标记图像
        cv2.imwrite(os.path.join(ROOT_DIR, 'original_marked_digits.png'), original_marked_img)
        log_message("原始尺寸标记后的图像已保存到 original_marked_digits.png")
        
        # 更新结果
        results['recognized_digits'] = unique_digits
        results['min_value'] = min_value
        results['min_digits'] = min_digits
        
        # 在控制台用ASCII艺术展示最小数字
        log_message("\n最小数字的ASCII表示:")
        # 创建一个简单的数字ASCII表示
        ascii_digits = {
            0: ["###", "# #", "# #", "# #", "###"],
            1: ["  #", "  #", "  #", "  #", "  #"],
            2: ["###", "  #", "###", "#  ", "###"],
            3: ["###", "  #", "###", "  #", "###"],
            4: ["# #", "# #", "###", "  #", "  #"],
            5: ["###", "#  ", "###", "  #", "###"],
            6: ["###", "#  ", "###", "# #", "###"],
            7: ["###", "  #", "  #", "  #", "  #"],
            8: ["###", "# #", "###", "# #", "###"],
            9: ["###", "# #", "###", "  #", "###"]
        }
        if min_value in ascii_digits:
            for line in ascii_digits[min_value]:
                log_message(line)
    else:
        log_message("未识别到任何数字")
        log_message("调试提示:")
        log_message("1. 确保屏幕上有清晰可见的数字")
        log_message("2. 数字最好使用黑色字体，白色背景")
        log_message("3. 避免屏幕上有过多干扰元素")
        log_message("4. 可以查看 osu 目录下的调试图像来分析问题")
    
    # 关闭日志文件
    log_file.close()

# 使用训练好的模型进行屏幕识别
try:
    time.sleep(5)
    # 检查是否存在测试图像
    test_image = os.path.join(ROOT_DIR, "test_digits.png")
    
    # 选择要使用的识别模式：1=标准识别，2=缩放识别
    recognition_mode = 2  # 默认使用缩放识别模式
    
    if recognition_mode == 1:
        # 使用标准数字识别
        print("使用标准数字识别模式")
        if os.path.exists(test_image):
            print(f"发现测试图像 {test_image}，将使用该图像进行测试")
            print("提示：您可以创建一张包含数字的图像并保存为 test_digits.png 来测试识别功能")
            results = recognize_screen_digits(model, transform, device, test_image_path=test_image)
        else:
            print("未发现测试图像，将使用屏幕截图")
            print("提示：您可以创建一张包含数字的图像并保存为 test_digits.png 来测试识别功能")
            results = recognize_screen_digits(model, transform, device)
    else:
        # 使用缩放数字识别（对缩小10倍的屏幕截图进行目标检测，并返回放大10倍后的坐标）
        print("使用缩放数字识别模式")
        print("将对屏幕截图缩小10倍进行处理，然后返回原始比例的坐标")
        
        if os.path.exists(test_image):
            print(f"发现测试图像 {test_image}，将使用该图像进行测试")
            results = recognize_scaled_digits(model, transform, device, test_image_path=test_image, scale_factor=5)
        else:
            print("未发现测试图像，将使用屏幕截图")
            results = recognize_scaled_digits(model, transform, device, scale_factor=5)
    
    # 显示识别结果摘要
    print("\n===== 识别结果摘要 =====")
    if results and 'min_value' in results and results['min_value'] is not None:
        print(f"最小数字: {results['min_value']}")
        print(f"识别到的数字数量: {len(results['recognized_digits'])}")
        
        if 'min_digits' in results and results['min_digits']:
            print("\n最小数字的原始比例坐标:")
            for digit, x, y, w, h, confidence in results['min_digits']:
                print(f"  数字 {digit} 在位置 ({x}, {y})，大小 {w}x{h}")
                print(f"  中心点坐标: ({x + w//2}, {y + h//2})")
                
        # 如果是缩放识别模式，显示缩放信息
        if 'scale_factor' in results:
            print(f"\n缩放信息:")
            print(f"  缩放因子: {results['scale_factor']}x")
            print(f"  原始图像尺寸: {results['original_image_size'][0]}x{results['original_image_size'][1]}")
            print(f"  处理后图像尺寸: {results['scaled_image_size'][0]}x{results['scaled_image_size'][1]}")
    else:
        print("未识别到任何数字")
except Exception as e:
    print(f"屏幕识别过程中出错: {e}")
    print("请确保已安装必要的库: pip install opencv-python pillow")
    import traceback
    traceback.print_exc()

# 将调试日志直接添加到文件开头，确保程序启动时就显示
print("=== OSU自动识别程序初始化 ===")
print(f"pyautogui库已加载: {pyautogui.__version__}")
try:
    # 测试pyautogui基本功能
    screen_size = pyautogui.size()
    print(f"屏幕尺寸: {screen_size}")
    print("pyautogui功能测试通过")
except Exception as e:
    print(f"警告: pyautogui功能测试失败: {e}")
    traceback.print_exc()

# 在主循环前再添加一次调试信息
# 同时修改run_model函数，确保日志输出正确

def run_model(model):
    print("\n=== 开始运行模型 ===")
    print("识别流程: 截图 -> 数字识别 -> 寻找最小数字 -> 鼠标点击")
    
    consecutive_failures = 0
    max_failures = 5
    last_successful_results = None  # 保存上次成功的结果作为回退
    
    while True:
        try:
            print("\n[步骤1] 正在截取屏幕并识别数字...")
            # 直接调用recognize_scaled_digits并捕获可能的异常
            results = None
            try:
                results = recognize_scaled_digits(model, transform, device, scale_factor=5)
                
                # 检查结果是否有效
                if results is not None:
                    consecutive_failures = 0
                    print("✅ 识别函数返回成功")
                    # 检查必要的键是否存在
                    if 'min_value' in results and 'min_digits' in results:
                        last_successful_results = results  # 保存成功的结果
                    else:
                        print("⚠️  识别结果缺少必要的键")
                        if last_successful_results:
                            print("🔄 使用上次成功的结果作为回退")
                            results = last_successful_results
                        else:
                            results = None
                else:
                    consecutive_failures += 1
                    print(f"❌ 警告: recognize_scaled_digits返回None! (连续失败: {consecutive_failures})")
                    # 使用上次成功的结果作为回退
                    if last_successful_results:
                        print("🔄 使用上次成功的结果作为回退")
                        results = last_successful_results
                    elif consecutive_failures >= max_failures:
                        print("⚠️  连续识别失败次数过多，等待2秒后重试...")
                        time.sleep(2)
                        consecutive_failures = 0
            except Exception as recog_error:
                consecutive_failures += 1
                print(f"❌ 识别过程出错: {recog_error}")
                traceback.print_exc()
                # 尝试使用上次成功的结果
                if last_successful_results:
                    print("🔄 使用上次成功的结果作为回退")
                    results = last_successful_results
            
            # 详细检查results内容
            if results is not None:
                print(f"[步骤2] 识别结果分析: {list(results.keys())}")
                
                if 'min_value' in results and results['min_value'] is not None:
                    print(f"  - 找到最小数字: {results['min_value']}")
                else:
                    print("  - 未找到最小数字值")
                    results = None
                
                if results and 'min_digits' in results and results['min_digits']:
                    print(f"  - 最小数字实例数量: {len(results['min_digits'])}")
                    print(f"  - 第一个实例: {results['min_digits'][0]}")
                elif results:
                    print("  - 最小数字实例列表为空")
                    results = None
            
            # 执行鼠标移动和点击
            if results:
                print("[步骤3] 选择最佳数字位置...")
                center_digit = None
                max_confidence = -1
                min_distance = float('inf')
                
                try:
                    screen_width, screen_height = pyautogui.size()
                    screen_center_x, screen_center_y = screen_width // 2, screen_height // 2
                    print(f"  - 屏幕中心位置: ({screen_center_x}, {screen_center_y})")
                except Exception as size_error:
                    print(f"❌ 获取屏幕尺寸失败: {size_error}")
                    time.sleep(1)
                    continue
                
                # 选择最佳数字
                for digit, x, y, w, h, confidence in results['min_digits']:
                    distance_to_center = ((x + w//2 - screen_center_x) ** 2 + 
                                         (y + h//2 - screen_center_y) ** 2) ** 0.5
                    
                    if confidence > max_confidence or (
                       confidence == max_confidence and distance_to_center < min_distance):
                        max_confidence = confidence
                        min_distance = distance_to_center
                        center_digit = (digit, x, y, w, h, confidence)
                
                if center_digit:
                    digit, x, y, w, h, confidence = center_digit
                    click_x = x + w // 2
                    click_y = y + h // 3
                    print(f"[步骤4] 执行鼠标操作:")
                    print(f"  - 目标位置: ({click_x}, {click_y})")
                    print(f"  - 数字: {digit}, 置信度: {confidence:.2f}")
                    
                    try:
                        # 先获取当前鼠标位置
                        current_pos = pyautogui.position()
                        print(f"  - 当前鼠标位置: {current_pos}")
                        
                        # 执行移动
                        print(f"  - 正在移动鼠标到目标位置...")
                        pyautogui.moveTo(click_x, click_y)
                        
                        # 验证移动是否成功
                        new_pos = pyautogui.position()
                        print(f"  - 移动后鼠标位置: {new_pos}")
                        
                        # 执行点击
                        print(f"  - 执行点击...")
                        pyautogui.click()
                        print(f"✅ 点击成功完成")
                    except Exception as mouse_error:
                        print(f"❌ 鼠标操作失败: {mouse_error}")
                        traceback.print_exc()
                        # 检查是否是权限问题
                        print("💡 可能的解决方案: 确保Python有足够的权限控制鼠标，或者关闭游戏的安全模式")
                else:
                    print("⚠️  未找到符合条件的数字位置")
            else:
                print("⚠️  没有有效的结果可供处理")
            
            # 等待间隔
            wait_time = 0.1 if results else 0.5  # 识别成功时短等待，失败时长等待
            print(f"[完成] 等待 {wait_time} 秒后继续...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n用户中断了程序")
            break
        except Exception as e:
            print(f"\n❌ 运行循环出错: {e}")
            traceback.print_exc()
            time.sleep(1)
            continue

print("\n程序初始化完成，准备接收命令")

# 恢复主循环，确保程序能正常接收命令
while True:
    cmd = input("请输入命令 (1: 神經網絡玩osu): ")
    if cmd == "1":
        run_model(model)
    elif cmd == "exit":
        break