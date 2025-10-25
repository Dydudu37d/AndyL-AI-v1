import queue
import time
import logging
import threading
from typing import Optional, Callable
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SpeechRecognizer")

# 检查是否安装了SpeechRecognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("SpeechRecognition未安装，语音识别功能将不可用")

class SpeechRecognizer:
    """语音识别器"""
    
    def __init__(self):
        self.is_listening = False
        self.recognition_thread = None
        self.callback = None
        self.recognizer = None
        self.microphone = None
        
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self._adjust_for_ambient_noise()
            except Exception as e:
                logger.error(f"初始化语音识别失败: {e}")
                SPEECH_RECOGNITION_AVAILABLE = False
    
    def _adjust_for_ambient_noise(self):
        """调整环境噪声"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return
            
        try:
            logger.info("正在调整环境噪声，请保持安静...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("环境噪声调整完成")
        except Exception as e:
            logger.error(f"调整环境噪声失败: {e}")
    
    def recognize_speech(self, audio_data) -> Optional[str]:
        """识别语音"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.error("语音识别功能不可用")
            return None
            
        try:
            # 首先尝试Google语音识别
            text = self.recognizer.recognize_google(audio_data, language='zh-CN')
            return text
        except sr.UnknownValueError:
            logger.warning("无法识别语音")
            return None
        except sr.RequestError as e:
            logger.error(f"语音识别服务错误: {e}")
            return None
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return None
    
    def listen_once(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """单次监听"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.error("语音识别功能不可用")
            return None
            
        try:
            with self.microphone as source:
                logger.info("正在聆听...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            text = self.recognize_speech(audio)
            if text:
                logger.info(f"识别结果: {text}")
            return text
            
        except sr.WaitTimeoutError:
            logger.info("聆听超时")
            return None
        except Exception as e:
            logger.error(f"监听失败: {e}")
            return None
    
    def _recognition_worker(self):
        """语音识别工作线程"""
        while self.is_listening:
            try:
                text = self.listen_once(timeout=1, phrase_time_limit=8)
                if text and self.callback:
                    self.callback(text)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"识别工作线程错误: {e}")
                time.sleep(1)
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始持续监听"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.error("语音识别功能不可用，请安装SpeechRecognition")
            return
        
        if self.is_listening:
            logger.warning("已经在监听中")
            return
        
        self.callback = callback
        self.is_listening = True
        self.recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.recognition_thread.start()
        logger.info("开始持续监听")
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2)
        logger.info("停止监听")
    
    def get_microphone_list(self) -> list:
        """获取可用的麦克风列表"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return []
            
        try:
            return sr.Microphone.list_microphone_names()
        except Exception as e:
            logger.error(f"获取麦克风列表失败: {e}")
            return []
    
    def set_microphone(self, device_index: int):
        """设置麦克风设备"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return False
            
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            self._adjust_for_ambient_noise()
            logger.info(f"已切换到麦克风设备 {device_index}")
            return True
        except Exception as e:
            logger.error(f"设置麦克风失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查语音识别是否可用"""
        return SPEECH_RECOGNITION_AVAILABLE


class DummySpeechRecognizer:
    """虚拟语音识别器（用于测试）"""
    
    def __init__(self):
        self.is_listening = False
        self.callback = None
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始监听"""
        logger.warning("语音识别功能不可用，使用虚拟模式")
        self.callback = callback
        self.is_listening = True
        
        # 创建一个模拟输入的线程
        def input_thread():
            while self.is_listening:
                try:
                    text = input("请输入模拟语音输入（或输入'stop'停止）: ")
                    if text.lower() == 'stop':
                        self.stop_listening()
                        break
                    if text and self.callback:
                        self.callback(text)
                except:
                    break
        
        threading.Thread(target=input_thread, daemon=True).start()
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
    
    def is_available(self) -> bool:
        return False


# 单例实例
speech_recognizer_instance = None

def get_speech_recognizer():
    """获取语音识别器实例"""
    global speech_recognizer_instance
    if speech_recognizer_instance is None:
        try:
            speech_recognizer_instance = SpeechRecognizer()
            if not speech_recognizer_instance.is_available():
                logger.warning("语音识别不可用，使用虚拟模式")
                speech_recognizer_instance = DummySpeechRecognizer()
        except Exception as e:
            logger.error(f"创建语音识别器失败: {e}")
            speech_recognizer_instance = DummySpeechRecognizer()
    return speech_recognizer_instance