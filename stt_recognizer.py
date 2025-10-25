import logging
import threading
import time
from typing import Optional, Callable
from stt import recognize_speech

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("STTSpeechRecognizer")

class STTSpeechRecognizer:
    """基于stt.py的语音识别器封装"""
    
    def __init__(self):
        self.is_listening = False
        self.callback = None
        self.listener_thread = None
        self.continuous_listening = False
    
    def initialize(self) -> bool:
        """初始化语音识别器"""
        try:
            # stt.py没有复杂的初始化过程，这里主要是确认依赖已安装
            logger.info("STT语音识别器初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化STT语音识别器失败: {e}")
            return False
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始监听语音输入"""
        if not self.initialize():
            logger.error("STT语音识别器初始化失败，无法开始监听")
            return
        
        self.callback = callback
        self.is_listening = True
        self.continuous_listening = True
        
        logger.info("STT语音识别已启动")
        print("STT语音识别已启动，请说话...（说'停止监听'可以退出）")
        
        # 启动监听线程
        self.listener_thread = threading.Thread(target=self._listen_thread, daemon=True)
        self.listener_thread.start()
    
    def start_realtime_listening(self, callback: Callable[[str], None]):
        """启动实时监听（与start_listening相同，为了兼容性）"""
        self.start_listening(callback)
    
    def _listen_thread(self):
        """监听线程"""
        try:
            # 调用stt.py中的识别函数
            text = recognize_speech()
            
            if text and self.callback:
                self.callback(text)
                
        except Exception as e:
            logger.error(f"监听线程错误: {e}")
            time.sleep(0.5)
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        self.continuous_listening = False
        
        if self.listener_thread:
            self.listener_thread.join(timeout=2.0)  # 等待线程结束，最多等待2秒
            
        logger.info("STT语音识别已停止")
        print("语音识别已停止")

# 单例模式
_speech_recognizer_instance = None

def get_speech_recognizer(stt_type=None):
    """获取语音识别器实例（为了与其他识别器保持接口一致）"""
    global _speech_recognizer_instance
    if _speech_recognizer_instance is None:
        _speech_recognizer_instance = STTSpeechRecognizer()
    return _speech_recognizer_instance