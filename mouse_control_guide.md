# 鼠标控制模块使用指南

## 概述

鼠标控制模块是AI VTuber系统的一个重要组成部分，允许AI根据屏幕内容和分析结果执行鼠标操作。本指南详细介绍了该模块的功能、使用方法和故障排除。

## 核心组件

### MouseController类

`mouse_controller.py`文件中的`MouseController`类是鼠标控制的核心组件，提供了以下功能：

- 鼠标移动（绝对坐标和相对坐标）
- 鼠标点击（左键、右键、双击）
- 鼠标滚轮操作
- 键盘按键模拟

### AIControlExecutor类

`ai_control_executor.py`文件中的`AIControlExecutor`类负责解析和执行AI生成的命令：

- 解析m_、k_、t_等前缀的命令
- 根据命令类型调用相应的鼠标或键盘控制方法
- 实现命令队列和执行逻辑

### AIComputerController类

`ai_computer_controller.py`文件中的`AIComputerController`类是整个控制流程的协调者：

- 捕获屏幕内容
- 发送给AI进行分析
- 解析AI响应中的控制命令
- 调用AIControlExecutor执行命令

## 命令格式

系统支持以下格式的鼠标控制命令：

### 1. 基础移动命令

- `m_down`：鼠标向下移动
- `m_up`：鼠标向上移动
- `m_left`：鼠标向左移动
- `m_right`：鼠标向右移动

### 2. 点击命令

- `m_click`：鼠标左键点击
- `m_double_click`：鼠标左键双击
- `m_right_click`：鼠标右键点击

### 3. 滚轮命令

- `m_scroll_up`：鼠标滚轮向上滚动
- `m_scroll_down`：鼠标滚轮向下滚动

### 4. 绝对位置移动

- `m_move_x_y`：将鼠标移动到绝对坐标(x,y)，例如 `m_move_500_300`

### 5. 命令组合

命令可以通过竖线(`|`)组合在一起，例如：
- `m_right|m_down|m_click`：先向右移动，再向下移动，然后点击
- `m_move_500_300|m_click`：移动到指定坐标后点击

## 支持的命令格式

系统可以识别多种格式的命令输出，包括：

1. 标准格式：
   ```
   m_down|m_up|m_left|m_right
   ```

2. Markdown代码块格式：
   ```
   ```
   m_down|m_up|m_left|m_right
   ```
   ```

3. 带有前缀的格式：
   ```
   <|FunctionCallBegin|>m_down|m_up|m_left|m_right<|FunctionCallEnd|>
   ```

4. 引号包围的格式：
   ```
   "m_down|m_up|m_left|m_right"
   ```

## 配置选项

鼠标控制模块的主要配置选项包括：

### 环境变量配置

```env
# AI类型配置（ollama或localai）
AI_TYPE=ollama

# Ollama配置
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2:latest

# LocalAI配置
LOCALAI_HOST=localhost
LOCALAI_PORT=8080
LOCALAI_MODEL=gpt-3.5-turbo

# 截图目录
SCREENSHOT_DIR=screenshots
```

### 鼠标移动参数

在`AIControlExecutor`类中，可以调整以下参数：

- `mouse_move_step`：鼠标每次移动的步长（像素）
- `smooth_movement`：是否启用平滑移动效果
- `movement_delay`：移动步骤之间的延迟（秒）

## 使用方法

### 基本使用

```python
from mouse_controller import MouseController
from ai_control_executor import AIControlExecutor

# 创建鼠标控制器实例
mouse_controller = MouseController()

# 直接使用鼠标控制器
mouse_controller.move_mouse(100, 100, absolute=True)
mouse_controller.click()

# 或者使用AI控制执行器
control_executor = AIControlExecutor()
control_executor.execute_command("m_move_500_300|m_click")
```

### 集成到主系统

```python
from ai_computer_controller import AIComputerController

# 创建AI电脑控制器
controller = AIComputerController()

# 执行单轮控制
controller.run_single_cycle()

# 或者连续控制
controller.run_continuous()
```

## 测试工具

项目包含多个测试工具，用于验证鼠标控制功能：

### 测试鼠标移动功能

```bash
python test_mouse_movement.py
```

这个脚本会测试以下内容：
- 直接使用pynput库控制鼠标
- 使用MouseController类控制鼠标
- 通过AIControlExecutor执行命令控制鼠标

### 测试完整AI控制流程

```bash
python test_ai_control_flow_complete.py
```

这个脚本会模拟完整的AI控制流程：
- 生成模拟AI响应
- 解析命令
- 执行鼠标操作

### 监控AI响应日志

```bash
python check_ai_responses.py
```

这个工具可以帮助你监控和分析AI生成的响应，特别是命令格式是否正确。

## 问题排查

### 鼠标无法移动的解决方法

如果遇到鼠标无法移动但键盘正常工作的问题，可以尝试以下解决方法：

1. **使用修复版本的控制器**
   ```bash
   # 备份原文件
   copy ai_computer_controller.py ai_computer_controller.bak
   
   # 使用修复版本
   copy ai_computer_controller_fixed.py ai_computer_controller.py
   ```

2. **检查命令格式**
   使用`check_ai_responses.py`工具监控AI生成的响应，确保命令格式正确。

3. **验证pynput库安装**
   ```bash
   python -c "import pynput; print('pynput库已安装')"
   
   # 如果需要重新安装
   pip install --upgrade pynput
   ```

4. **检查系统权限**
   - 以管理员身份运行程序
   - 检查防病毒软件是否阻止了程序的鼠标控制功能

5. **查看日志文件**
   检查`ai_computer_control.log`文件中的错误信息。

### 常见问题解答

**Q: 为什么键盘可以工作，但鼠标不行？**

A: 这通常是因为命令解析问题，系统能够识别键盘命令（k_前缀），但无法正确解析鼠标命令（m_前缀）。使用修复版本的控制器通常可以解决这个问题。

**Q: 如何调整鼠标移动速度？**

A: 可以修改`AIControlExecutor`类中的`mouse_move_step`参数，增大该值可以加快鼠标移动速度。

**Q: 如何查看AI生成的原始命令？**

A: 使用`check_ai_responses.py`工具可以实时监控和分析AI生成的响应。

## 注意事项

1. 请确保程序有足够的权限控制鼠标，特别是在Windows系统上
2. 在某些安全软件或游戏环境中，可能会限制程序对鼠标的控制
3. 定期检查日志文件，以便及时发现和解决问题
4. 使用该模块时请小心，避免意外操作导致的数据丢失