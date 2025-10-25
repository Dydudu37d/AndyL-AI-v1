import os
import logging
import threading
import tempfile
import pygame
import requests
import json
import time
import asyncio
import socket
import comtypes.client  # 用于Windows语音API
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
from ai_brain import OllamaDeepSeekClient, AIBrain, VTuberPersona
from obs import OBSController
from dotenv import load_dotenv
import edge_tts  # 用于Edge TTS
from interactive_obs_text_tool import *

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTSSpeaker")

def get_local_ip():
    try:
        # 建立一个临时的UDP连接到Google的DNS服务器
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        # 获取连接的本地IP地址
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return f"获取IP时出错: {e}"

local_ip = get_local_ip()
print(f"本机内网IP: {local_ip}")
host = input("请输入OBS WebSocket服务器地址 (默认: 192.168.0.186): ") or local_ip

# OBS控制器配置
obs_controller = OBSController(
    host=host,
    port=4455,
    password='gR7UXLWyqEBaRd2S'
)

textAI = OBSTextTool()

# 高级神经语音合成模型
class AdvancedVoiceModel(nn.Module):
    """改进的语音合成神经网络模型"""
    def __init__(self, embedding_dim=256, hidden_dim=512, output_dim=16000):
        super(AdvancedVoiceModel, self).__init__()
        # 文本嵌入层（假设使用简单的字符嵌入）
        self.char_embedding = nn.Embedding(10000, embedding_dim)  # 假设使用10000个字符的词汇表
        
        # LSTM层处理文本序列
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        
        # 解码器网络（将文本特征转换为音频特征）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)  # 输出音频样本
        )
        
        # 输出处理层
        self.output_processor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, text_features, sequence_lengths=None):
        # 处理文本嵌入
        embedded = self.char_embedding(text_features)
        
        # LSTM处理
        if sequence_lengths is not None:
            # 对变长序列进行pack操作
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths, batch_first=True, enforce_sorted=False)
            packed_output, (h_n, c_n) = self.lstm(packed_embedded)
            # 对输出进行unpack操作
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (h_n, c_n) = self.lstm(embedded)
        
        # 注意力机制
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # 全局池化，将序列特征转换为固定长度特征
        global_features = torch.mean(attn_output, dim=1)
        
        # 解码层生成音频特征
        audio_features = self.decoder(global_features)
        
        # 进一步处理音频特征以提高质量
        audio_features = audio_features.unsqueeze(1)  # 添加通道维度 [batch, 1, output_dim]
        processed_audio = self.output_processor(audio_features)
        processed_audio = processed_audio.squeeze(1)  # 移除通道维度
        
        return processed_audio

# 文本处理器
class TextProcessor:
    """处理文本输入，将其转换为模型可处理的特征"""
    def __init__(self):
        # 基本中文字符映射（实际应用中应该使用更完整的映射表）
        self.char_to_idx = self._build_basic_vocab()
        self.max_text_length = 200  # 最大文本长度
        
    def _build_basic_vocab(self):
        """构建基本词汇表"""
        # 基本ASCII字符
        vocab = {chr(i): i + 1 for i in range(32, 127)}  # 1-95
        
        # 添加常用中文字符（示例）
        common_chinese = "你好我是在这有个人们的一了不很这是个大中国北京上海广州深圳天南海北"
        start_idx = 100
        for char in common_chinese:
            if char not in vocab:
                vocab[char] = start_idx
                start_idx += 1
        
        # 添加特殊标记
        vocab['<UNK>'] = 0  # 未知字符
        vocab['<PAD>'] = start_idx  # 填充字符
        
        return vocab
    
    def text_to_features(self, text):
        """将文本转换为模型可处理的特征"""
        # 过滤文本
        filtered_text = self._filter_text(text)
        
        # 将文本转换为索引序列
        features = []
        for char in filtered_text:
            # 截取最大长度
            if len(features) >= self.max_text_length:
                break
            
            # 转换字符到索引
            if char in self.char_to_idx:
                features.append(self.char_to_idx[char])
            else:
                features.append(self.char_to_idx['<UNK>'])  # 未知字符
        
        # 填充到最大长度
        if len(features) < self.max_text_length:
            pad_length = self.max_text_length - len(features)
            features.extend([self.char_to_idx['<PAD>']] * pad_length)
        
        return torch.tensor(features, dtype=torch.long).unsqueeze(0)  # 添加batch维度
    
    def _filter_text(self, text):
        """过滤文本中的特殊字符"""
        # 保留中文字符、英文和基本标点
        filtered = re.sub(r'[^一-龥a-zA-Z0-9，。！？,.!?]', '', text)
        return filtered.strip()

# 音频处理器
class AudioProcessor:
    """处理音频数据，包括生成、增强和保存"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def process_audio(self, audio_data):
        """处理模型生成的音频数据"""
        # 确保音频数据形状正确
        if len(audio_data.shape) == 1:
            audio_data = audio_data.unsqueeze(0)  # 添加batch维度
        
        # 转换为numpy数组
        audio_np = audio_data.detach().cpu().numpy().flatten()
        
        # 归一化音频
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val
        
        # 添加简单的音频增强
        audio_np = self._enhance_audio(audio_np)
        
        # 转换为16位整数
        audio_np = np.int16(audio_np * 32767)
        
        return audio_np
    
    def _enhance_audio(self, audio_data):
        """简单的音频增强"""
        # 添加轻微的预加重（高频增强）
        pre_emphasis = 0.97
        enhanced = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # 轻微的音量归一化
        max_amplitude = np.max(np.abs(enhanced))
        if max_amplitude > 0:
            enhanced = enhanced * 0.9 / max_amplitude
        
        return enhanced
    
    def save_to_wav(self, audio_data, output_file):
        """保存音频数据为WAV文件"""
        import wave
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
            
        return output_file


class TTSSpeaker:
    """TTS语音合成器，支持WindowI和LocalAI TTS(已廢棄)"""
    
    def __init__(self):
        self.is_initialized = False
        self.voice_engine = None
        self.current_voice = None
        self._is_speaking = False
        self._lock = threading.Lock()
        self._tts_type = os.getenv("TTS_TYPE", "edge")  # windows, localai, edge 或 custom
        
        # LocalAI配置 (已廢棄)
        self.localai_host = os.getenv("LOCALAI_HOST", "localhost")
        self.localai_port = os.getenv("LOCALAI_PORT", "8080")
        self.localai_tts_model = os.getenv("LOCALAI_TTS_MODEL", "tts-1")
        self.localai_tts_voice = os.getenv("LOCALAI_TTS_VOICE", "alloy")
        
        # Edge TTS配置
        self.edge_tts_voice = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
        self.edge_tts_rate = os.getenv("EDGE_TTS_RATE", "+0%")
        self.edge_tts_volume = os.getenv("EDGE_TTS_VOLUME", "+0%")
        
        # 自定义模型配置
        self.custom_model_path = os.getenv("CUSTOM_TTS_MODEL_PATH", "G:\AndyL AI v1\AndyL_AI_speak_v1\exported_voice_models\andyL_voice_model_latest.pth")
        self.custom_model = None
        self.system_prompt = AIBrain()
    
    # 初始化pygame用于播放音频
    try:
        pygame.mixer.init()
    except pygame.error as e:
        logger.warning(f"初始化pygame失败: {e}")
    
    def initialize(self, voice_id: str = "zh-CN-XiaoxiaoNeural") -> bool:
        """初始化TTS引擎"""
        with self._lock:
            try:
                if self._tts_type == "localai":
                    # 使用LocalAI的TTS服务
                    logger.info(f"初始化LocalAI TTS，模型: {self.localai_tts_model}")
                    # 简单测试连接
                    test_result = self._test_localai_connection()
                    if test_result:
                        self.is_initialized = True
                        self.current_voice = voice_id
                        logger.info("LocalAI TTS初始化成功")
                        return True
                    else:
                        logger.error("LocalAI TTS初始化失败，尝试使用Windows语音API")
                        self._tts_type = "windows"
                elif self._tts_type == "edge":
                    # 使用Edge TTS服务
                    logger.info(f"初始化Edge TTS，语音: {voice_id or self.edge_tts_voice}")
                    # 设置Edge TTS语音
                    self.edge_tts_voice = voice_id or self.edge_tts_voice
                    self.is_initialized = True
                    self.current_voice = self.edge_tts_voice
                    logger.info("Edge TTS初始化成功")
                    return True
                
                # 默认使用Windows语音API
                logger.info(f"初始化Windows语音API，语音: {voice_id}")
                self.voice_engine = comtypes.client.CreateObject("SAPI.SpVoice")
                
                # 设置语音
                voices = self.voice_engine.GetVoices()
                for i in range(voices.Count):
                    voice = voices.Item(i)
                    if voice_id in voice.GetDescription() or i == 0:
                        self.voice_engine.Voice = voice
                        self.current_voice = voice.GetDescription()
                        break
                
                self.is_initialized = True
                logger.info(f"TTS初始化成功，当前语音: {self.current_voice}")
                return True
            except Exception as e:
                logger.error(f"TTS初始化失败: {e}")
                self.is_initialized = False
                return False
    
    def _test_localai_connection(self) -> bool:
        """测试LocalAI TTS服务连接"""
        try:
            url = f"http://{self.localai_host}:{self.localai_port}/v1/audio/speech"
            headers = {'Content-Type': 'application/json'}
            data = {
                "model": self.localai_tts_model,
                "input": "测试",
                "voice": self.localai_tts_voice
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LocalAI连接测试失败: {e}")
            return False
    
    def filter_emojis_and_special_chars(self, text: str) -> str:
        """过滤文本中的表情符号和特殊字符，但保留中文字符、英文和其他正常文本"""
        logger.debug(f"过滤前文本长度: {len(text)}字符，内容: {text}")
        
        # 安全版本: 使用直接替换的方式过滤表情符号
        # 列出常见的表情符号进行过滤
        filtered_text = text
        
        # 过滤表情符号（通过直接替换而不是正则表达式范围）
        emojis_to_filter = "🙏😊😀😃😄😁😂😅😍🥰😘😗😙😚😋😛😝😜🍜👌🤪🤨🤔💪🎮🧐🤓😎🤩🥳😏😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬🤯😳🥵🥶😱😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😬🙄😯😦😧😮😲😴😪🤤😵🤐🥴🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾🎉👋🤣🍕😆👍📺"
        for emoji in emojis_to_filter:
            filtered_text = filtered_text.replace(emoji, '')
        
        # 过滤零宽字符
        filtered_text = filtered_text.replace('+[+', '')  # 零宽连接符
        filtered_text = filtered_text.replace('\u200d', '')  # 零宽连接符
        filtered_text = filtered_text.replace('\u200b', '')  # 零宽空格
        filtered_text = filtered_text.replace('\u200c', '')  # 零宽非连接符
        filtered_text = filtered_text.replace('*', '')  # 星号
        filtered_text = filtered_text.replace('+]=+', '')  # 右方括号
        
        # 过滤think标记内容
        # 查找并移除以think开头和结尾的内容
        filtered_text = re.sub(r'think.*?think', '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
        
        # 只移除特定的可能影响TTS的标记，但保留正常的文本内容
        filtered_text = re.sub(r'\*\s*打卡\s*\*|\*\s*笑容开怀\s*\*', '', filtered_text)
        
        # 确保返回一个有效的文本
        result = filtered_text.strip()
        
        # 检查结果
        if not result:
            logger.warning(f"过滤后文本为空，原始文本: {text}")
            # 如果过滤后文本为空，返回原始文本的前50个字符（安全回退）
            return text[:50].strip()
        else:
            logger.debug(f"过滤后文本长度: {len(result)}字符，内容: {result}")

        # 保留中文字符、英文和基本标点
        filtered = re.sub(r'[^一-龥a-zA-Z0-9，。！？,.!?]', '', result)
        
        return filtered.strip()
    
    def speak(self, text: str) -> bool:
        """播放文本（同步方式）"""
        if not self.is_initialized:
            logger.error("TTS未初始化")
            return False
        # 过滤表情符号和特殊字符
        filtered_text = self.filter_emojis_and_special_chars(text)
        logger.debug(f"原始文本: {text}")
        logger.debug(f"过滤后文本: {filtered_text}")
        obs_controller.set_text_source_content("textai123",filtered_text)
        obs_controller.set_text_source_content("textai",filtered_text)
        textAI.text("textai",filtered_text)
        textAI.text("textai123",filtered_text)
        
        with self._lock:
            self._is_speaking = True
            try:
                if self._tts_type == "localai":
                    return self._speak_with_localai(filtered_text)
                elif self._tts_type == "edge":
                    return self._speak_with_edge_tts(filtered_text)
                elif self._tts_type == "custom":
                    return self._speak_with_custom_model(filtered_text)
                else:
                    # 使用Windows语音API
                    logger.info(f"使用Windows语音API播放，文本长度: {len(filtered_text)}字符")
                    self.voice_engine.Speak(filtered_text)
                return True
            except Exception as e:
                logger.error(f"语音播放失败: {e}")
                # 尝试降级到模拟播放
                self._simulate_speech(filtered_text)
                return False
            finally:
                self._is_speaking = False
                obs_controller.set_text_source_content("textai123","")
                obs_controller.set_text_source_content("textai","")
                textAI.text("textai","")
                textAI.text("textai123","")
    
    def _speak_with_localai(self, text: str) -> bool:
        obs_controller.set_text_source_content("textai",text)
        """使用LocalAI进行语音合成并播放"""
        try:
            url = f"http://{self.localai_host}:{self.localai_port}/v1/audio/speech"
            headers = {'Content-Type': 'application/json'}
            data = {
                "model": self.localai_tts_model,
                "input": text,
                "voice": self.localai_tts_voice
            }
            
            # 发送请求获取语音
            logger.info(f"向LocalAI发送TTS请求，文本长度: {len(text)}字符")
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            
            if response.status_code == 200:
                # 保存临时音频文件并播放
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                try:
                    # 播放音频
                    pygame.mixer.music.load(temp_file_path)
                    pygame.mixer.music.play()
                    
                    # 等待播放完成
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    return True
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
            else:
                logger.error(f"LocalAI TTS请求失败: HTTP {response.status_code}, {response.text}")
                # 尝试降级到Windows语音API
                self._tts_type = "windows"
                try:
                    self.voice_engine = comtypes.client.CreateObject("SAPI.SpVoice")
                    self.voice_engine.Speak(text)
                    return True
                except:
                    # 最后降级到模拟播放
                    self._simulate_speech(text)
                    return False
        except Exception as e:
            logger.error(f"LocalAI TTS执行失败: {e}")
            # 尝试降级到模拟播放
            self._simulate_speech(text)
            return False
    
    def _speak_with_edge_tts(self, text: str) -> bool:
        obs_controller.set_text_source_content("textai123",text)
        """使用Edge TTS进行语音合成并播放"""
        try:
            # 创建临时文件保存音频
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # 使用Edge TTS生成音频
            logger.info(f"使用Edge TTS生成语音，文本长度: {len(text)}字符，语音: {self.edge_tts_voice}")
            
            # 调用Edge TTS合成语音
            communicate = edge_tts.Communicate(text, self.edge_tts_voice)
            logger.debug(f"Edge TTS配置: rate={self.edge_tts_rate}, volume={self.edge_tts_volume}")
            communicate.save_sync(temp_file_path)
            logger.info(f"Edge TTS音频生成成功，文件路径: {temp_file_path}")
            
            # 播放音频
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
        except Exception as e:
            logger.error(f"Edge TTS执行失败: {e}")
            # 尝试降级到Windows语音API
            self._tts_type = "windows"
            try:
                self.voice_engine = comtypes.client.CreateObject("SAPI.SpVoice")
                self.voice_engine.Speak(text)
                return True
            except:
                # 最后降级到模拟播放
                self._simulate_speech(text)
                return False
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    async def speak_async(self, text: str) -> threading.Thread:
        """异步播放文本"""
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()
        return thread
    
    def is_speaking_now(self) -> bool:
        """检查是否正在播放语音"""
        with self._lock:
            return self._is_speaking
    
    def stop(self):
        """停止当前播放"""
        with self._lock:
            if self._is_speaking:
                try:
                    if self._tts_type == "windows" and self.voice_engine:
                        self.voice_engine.Speak(None, 3)  # SVSFPurgeBeforeSpeak
                    elif pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    self._is_speaking = False
                    logger.info("TTS播放已停止")
                except Exception as e:
                    logger.error(f"停止播放时出错: {e}")
    
    def _simulate_speech(self, text: str):
        """模拟语音播放（当其他方法都失败时的回退方案）"""
        logger.warning("使用模拟语音播放")
        # 简单模拟语音播放的时间
        simulated_time = len(text) * 0.05  # 假设每个字符需要0.05秒
        time.sleep(min(simulated_time, 5))  # 最多模拟5秒
    
    def set_tts_type(self, tts_type: str):
        """设置TTS类型"""
        if tts_type in ["windows", "localai", "edge", "custom"]:
            with self._lock:
                self._tts_type = tts_type
                logger.info(f"TTS类型已切换为: {tts_type}")
                # 重新初始化
                if self.is_initialized:
                    if tts_type == "custom":
                        self.use_custom_voice_model()
                    else:
                        self.initialize(self.current_voice)
        else:
            logger.error(f"不支持的TTS类型: {tts_type}")
    
    def get_tts_type(self) -> str:
        """获取当前TTS类型"""
        return self._tts_type

# 单例实例
_tts_speaker_instance = None


def get_tts_speaker() -> TTSSpeaker:
    """获取TTS说话器实例（单例模式）"""
    global _tts_speaker_instance
    if _tts_speaker_instance is None:
        _tts_speaker_instance = TTSSpeaker()
    return _tts_speaker_instance


# 测试代码（当直接运行此文件时执行）
if __name__ == "__main__":
    print("测试TTS说话器...")
    
    # 获取TTS实例
    tts = get_tts_speaker()
    
    # 初始化TTS
    initialized = tts.initialize("zh-CN-XiaoxiaoNeural")
    
    if initialized:
        print(f"TTS初始化成功，当前使用: {tts.get_tts_type()}")
        
        # 测试文本
        test_text = "你好，这是一个TTS测试。我可以将文本转换为语音。"
        
        # 测试同步播放
        print("测试同步播放...")
        tts.speak(test_text)
        
        # 测试异步播放
        print("测试异步播放...")
        asyncio.run(tts.speak_async("这是异步播放测试。"))
        
        # 等待异步播放完成
        time.sleep(3)
        
        # 如果当前不是Edge TTS，尝试切换到Edge TTS
        if tts.get_tts_type() != "edge":
            print("\n尝试切换到Edge TTS...")
            tts.set_tts_type("edge")
            if tts.is_initialized:
                print("Edge TTS初始化成功")
                tts.speak("这是使用Edge TTS生成的语音。")
            else:
                print("Edge TTS初始化失败，保持使用当前TTS")
        
        # 如果当前不是LocalAI TTS，尝试切换到LocalAI TTS
        if tts.get_tts_type() != "localai" and tts.get_tts_type() != "edge":
            print("\n尝试切换到LocalAI TTS...")
            tts.set_tts_type("localai")
            if tts.is_initialized:
                print("LocalAI TTS初始化成功")
                tts.speak("这是使用LocalAI生成的语音。")
            else:
                print("LocalAI TTS初始化失败，保持使用当前TTS")
        
        # 尝试切换到自定义TTS（如果支持）
        print("\n尝试切换到自定义TTS...")
        tts.set_tts_type("custom")
        if tts.is_initialized:
            print("自定义TTS初始化成功")
            tts.speak("这是使用自定义神经网络模型生成的语音。")
        else:
            print("自定义TTS初始化失败，可能是因为找不到模型文件或模型加载失败")
    else:
        print("TTS初始化失败")