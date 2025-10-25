# -*- coding: utf-8 -*-
"""
AI想法显示模块
这个模块提供了显示AI思考过程的功能，可以将AI的想法显示在OBS文本源上。
"""
import logging
import os
from tts_speaker import get_tts_speaker
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AI_Thought_Display")
"""
tts_speaker = get_tts_speaker()
tts_type = "edge"
"""

class AIThoughtDisplay:
    """AI想法显示管理器"""
    
    def __init__(self, text_tool):
        """初始化AI想法显示管理器"""
        self.text_tool = text_tool
        self.thought_source_name = "textai123"  # 默认的文本源名称
        self.connected = False
        self.max_thought_length = 10000  # 最大想法长度
        
    def set_text_source(self, source_name: str) -> bool:
        """设置用于显示想法的文本源"""
        self.thought_source_name = source_name
        logger.info(f"已设置想法显示文本源: {source_name}")
        return True
        
    def display_thought(self, thought: str, clear_after: Optional[int] = None) -> bool:
        """显示AI的想法
        
        参数:
            thought: AI的想法文本
            clear_after: 显示后多少秒清除，为None时不自动清除
            
        返回:
            是否显示成功
        """
        if not thought:
            logger.warning("没有想法可显示")
            return False
        
        # 限制想法长度
        #if len(thought) > self.max_thought_length:
            #thought = thought[:self.max_thought_length] + "..."
        
        # 在控制台打印AI想法
        print(f"[AI想法] {thought}")
        
        # 在OBS文本源上显示想法
        try:
            # 确保文本工具已连接
            if not self.text_tool.connected:
                self.text_tool.connect()
                self.connected = self.text_tool.connected
                
            if self.connected:
                success = self.text_tool.controller.set_text_source_content(
                    self.thought_source_name, 
                    f"{thought}"
                )
                
                if success:
                    logger.info(f"成功显示AI想法: {thought}")
                    
                    # 如果设置了清除时间，启动一个线程来清除
                    if clear_after and isinstance(clear_after, int) and clear_after > 0:
                        import threading
                        import time
                        
                        def clear_thought():
                            time.sleep(clear_after)
                            self.clear_thought()
                            
                        threading.Thread(target=clear_thought, daemon=True).start()
                    
                    return True
                else:
                    logger.warning(f"无法在OBS中显示想法")
                    return False
            else:
                logger.warning("未连接到OBS，仅在控制台显示想法")
                return True  # 即使OBS不可用，也认为在控制台显示成功
        except Exception as e:
            logger.error(f"显示AI想法时发生错误: {e}")
            # 即使出错，也认为在控制台显示成功
            return True
            
    def clear_thought(self) -> bool:
        """清除当前显示的AI想法"""
        try:
            if self.connected:
                success = self.text_tool.controller.set_text_source_content(
                    self.thought_source_name, ""
                )
                if success:
                    logger.info("已清除AI想法显示")
                    return True
            return False
        except Exception as e:
            logger.error(f"清除AI想法时发生错误: {e}")
            return False
            
    def format_thought(self, thought: str) -> str:
        """格式化AI想法文本"""
        # 去除多余的空格和换行
        formatted = ' '.join(thought.split())
        
        # 处理特殊字符（如表情符号）
        # 这里可以添加更多的格式化逻辑
        
        return formatted

# 单例实例
ai_thought_display_instance = None

def get_ai_thought_display(text_tool=None):
    """获取AI想法显示管理器实例（单例模式）"""
    global ai_thought_display_instance
    if ai_thought_display_instance is None:
        if text_tool:
            ai_thought_display_instance = AIThoughtDisplay(text_tool)
        else:
            # 如果没有提供text_tool，创建一个默认的
            logger.warning("没有提供text_tool，尝试创建默认的OBSTextTool实例")
            try:
                from interactive_obs_text_tool import OBSTextTool
                text_tool = OBSTextTool()
                ai_thought_display_instance = AIThoughtDisplay(text_tool)
            except ImportError:
                logger.error("无法导入OBSTextTool，AI想法显示功能可能受限")
                ai_thought_display_instance = AIThoughtDisplay(None)
    return ai_thought_display_instance