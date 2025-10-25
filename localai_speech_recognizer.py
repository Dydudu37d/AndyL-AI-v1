import os
import sys
import logging
import threading
import time
import wave
import pyaudio
import requests
import json
from typing import Optional, Callable
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LocalAISpeechRecognizer")


class LocalAISpeechRecognizer:
    """基于LocalAI的语音识别器"""
    
    def __init__(self):
        self.is_listening = False
        self.callback = None
        self.listener_thread = None
        self.audio_recorder = None
        
        # LocalAI配置
        self.localai_host = os.getenv("LOCALAI_HOST", "localhost")
        self.localai_port = os.getenv("LOCALAI_PORT", "8080")
        self.localai_stt_model = os.getenv("LOCALAI_STT_MODEL", "whisper-1")
        self.language = os.getenv("LANGUAGE", "zh")
        
        # 音频配置
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.silence_threshold = 300  # 静默阈值
        self.silence_duration = 2  # 静默持续时间（秒）用于检测语音结束
    
    def initialize(self) -> bool:
        """初始化语音识别器"""
        try:
            # 测试LocalAI连接
            if not self._test_localai_connection():
                logger.error("无法连接到LocalAI服务")
                return False
            
            # 初始化PyAudio
            self.audio_recorder = pyaudio.PyAudio()
            
            logger.info(f"LocalAI语音识别初始化成功，模型: {self.localai_stt_model}")
            return True
        except Exception as e:
            logger.error(f"初始化LocalAI语音识别失败: {e}")
            return False
    
    def _test_localai_connection(self) -> bool:
        """测试LocalAI连接"""
        try:
            url = f"http://{self.localai_host}:{self.localai_port}/v1/models"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LocalAI连接测试失败: {e}")
            return False
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始监听语音输入"""
        if not self.initialize():
            logger.error("LocalAI语音识别初始化失败，无法开始监听")
            self._start_fallback_mode(callback)
            return
        
        self.callback = callback
        self.is_listening = True
        
        logger.info("LocalAI语音识别已启动")
        print("LocalAI语音识别已启动，请说话...（说'停止监听'可以退出）")
        
        # 启动监听线程
        self.listener_thread = threading.Thread(target=self._listen_thread, daemon=True)
        self.listener_thread.start()
    
    def _listen_thread(self):
        """监听线程"""
        while self.is_listening:
            try:
                # 录制音频直到检测到静音
                audio_data = self._record_until_silence()
                
                if audio_data and len(audio_data) > 0:
                    # 将音频数据发送到LocalAI进行识别
                    text = self._recognize_speech(audio_data)
                    
                    if text and self.callback:
                        self.callback(text)
                        
                        # 检查是否需要停止监听
                        if '停止监听' in text or '停止' in text:
                            self.stop_listening()
                            break
            except Exception as e:
                logger.error(f"监听线程错误: {e}")
                time.sleep(0.5)
    
    def _record_until_silence(self):
        """录制音频直到检测到静音"""
        stream = None
        try:
            # 打开音频流
            stream = self.audio_recorder.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            frames = []
            silent_frames = 0
            speech_started = False
            
            while self.is_listening:
                data = stream.read(self.chunk)
                frames.append(data)
                
                # 计算音频能量（简单的音量检测）
                audio_energy = self._calculate_audio_energy(data)
                
                # 检测语音开始和结束
                if not speech_started and audio_energy > self.silence_threshold:
                    speech_started = True
                    logger.debug("检测到语音开始")
                elif speech_started:
                    if audio_energy < self.silence_threshold:
                        silent_frames += 1
                    else:
                        silent_frames = 0
                    
                    # 如果检测到足够长的静音，认为语音结束
                    if silent_frames > int(self.rate / self.chunk * self.silence_duration):
                        logger.debug("检测到语音结束")
                        break
            
            if not speech_started:
                return None
            
            return b''.join(frames)
        except Exception as e:
            logger.error(f"录制音频时出错: {e}")
            return None
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def _calculate_audio_energy(self, audio_data):
        """计算音频能量（用于检测静音）"""
        import numpy as np
        
        try:
            # 将音频数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算RMS（均方根）作为能量指标
            rms = np.sqrt(np.mean(np.square(audio_array)))
            return rms
        except Exception as e:
            logger.error(f"计算音频能量时出错: {e}")
            return 0
    
    def _recognize_speech(self, audio_data) -> Optional[str]:
        """将音频数据发送到LocalAI进行语音识别"""
        try:
            # 将音频数据保存到临时WAV文件
            with wave.open('temp_recording.wav', 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio_recorder.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
            
            # 准备发送到LocalAI的请求
            url = f"http://{self.localai_host}:{self.localai_port}/v1/audio/transcriptions"
            files = {
                'file': ('temp_recording.wav', open('temp_recording.wav', 'rb'), 'audio/wav')
            }
            data = {
                'model': self.localai_stt_model,
                'language': self.language
            }
            
            # 发送请求
            logger.info("向LocalAI发送语音识别请求")
            response = requests.post(url, files=files, data=data, timeout=30)
            
            # 清理临时文件
            os.remove('temp_recording.wav')
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '')
                logger.info(f"语音识别结果: {text}")
                return text
            else:
                logger.error(f"LocalAI语音识别请求失败: HTTP {response.status_code}, {response.text}")
                return None
        except Exception as e:
            logger.error(f"语音识别过程出错: {e}")
            # 尝试清理临时文件
            try:
                if os.path.exists('temp_recording.wav'):
                    os.remove('temp_recording.wav')
            except:
                pass
            return None
    
    def _start_fallback_mode(self, callback: Callable[[str], None]):
        """降级模式：使用文本输入"""
        logger.info("使用文本输入模式")
        self.callback = callback
        self.is_listening = True
        
        def input_thread():
            while self.is_listening:
                try:
                    text = input("语音识别不可用，请输入文本（输入'停止监听'退出）: ").strip()
                    if not text:
                        continue
                    
                    if text.lower() in ['停止监听', '停止', 'quit', 'exit']:
                        self.stop_listening()
                        break
                    
                    if self.callback:
                        self.callback(text)
                except Exception as e:
                    logger.error(f"输入线程错误: {e}")
                    break
        
        threading.Thread(target=input_thread, daemon=True).start()
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        
        # 等待监听线程结束
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        
        # 清理PyAudio资源
        if self.audio_recorder:
            try:
                self.audio_recorder.terminate()
            except:
                pass
        
        logger.info("LocalAI语音识别已停止")


# 创建一个新的工厂函数，结合Windows和LocalAI的语音识别器
def create_speech_recognizer(recognizer_type: str = "windows") -> object:
    """
    创建语音识别器
    recognizer_type: "windows" | "localai" | "stt"
    """
    if recognizer_type == "localai":
        try:
            from localai_speech_recognizer import LocalAISpeechRecognizer
            return LocalAISpeechRecognizer()
        except Exception as e:
            logger.error(f"创建LocalAI语音识别器失败: {e}")
            # 降级到Windows语音识别
            from windows_speech_recognizer import create_windows_speech_recognizer
            return create_windows_speech_recognizer()
    elif recognizer_type == "stt":
        try:
            from stt_recognizer import STTSpeechRecognizer
            return STTSpeechRecognizer()
        except Exception as e:
            logger.error(f"创建STT语音识别器失败: {e}")
            # 降级到Windows语音识别
            from windows_speech_recognizer import create_windows_speech_recognizer
            return create_windows_speech_recognizer()
    else:
        # 默认使用Windows语音识别
        from windows_speech_recognizer import create_windows_speech_recognizer
        return create_windows_speech_recognizer()


# 全局语音识别器实例
_global_speech_recognizer = None
_speech_recognizer_type = None


def get_speech_recognizer(recognizer_type: str = "stt") -> object:
    """
    获取语音识别器实例（支持Windows、LocalAI和STT）
    recognizer_type: "windows" | "localai" | "stt"
    """
    global _global_speech_recognizer, _speech_recognizer_type
    
    # 如果实例不存在，或者请求的类型与当前实例类型不同，则创建新实例
    if _global_speech_recognizer is None or _speech_recognizer_type != recognizer_type:
        _global_speech_recognizer = create_speech_recognizer(recognizer_type)
        _speech_recognizer_type = recognizer_type
    
    return _global_speech_recognizer


# 测试代码
if __name__ == "__main__":
    def test_callback(text):
        print(f"识别到: {text}")
    
    recognizer = LocalAISpeechRecognizer()
    print("开始测试LocalAI语音识别...")
    
    recognizer.start_listening(test_callback)
    
    try:
        # 保持主线程运行
        while recognizer.is_listening:
            time.sleep(1)
    except KeyboardInterrupt:
        recognizer.stop_listening()
        print("测试结束")