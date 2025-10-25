import threading
import time
import logging
import pythoncom
from typing import Optional, Callable
import win32com.client
import comtypes.client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WindowsSpeechRecognition")

class WindowsSpeechRecognizer:
    """Windows原生语音识别器"""
    
    def __init__(self):
        self.is_listening = False
        self.callback = None
        self.recognizer = None
        self.grammar = None
        self.recognizer_thread = None
        
    def initialize(self) -> bool:
        """初始化语音识别引擎"""
        try:
            # 初始化COM
            pythoncom.CoInitialize()
            
            # 创建语音识别引擎
            self.recognizer = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
            
            # 创建语音上下文
            context = self.recognizer.CreateRecoContext()
            
            # 创建语法
            self.grammar = context.CreateGrammar()
            self.grammar.DictationSetState(1)  # 启用听写模式
            
            # 注册事件处理
            from win32com.client import constants
            context.EventInterests = constants.SRERecognition | constants.SREFalseRecognition
            
            # 创建事件
            self.event = win32com.client.WithEvents(context, SpeechRecognitionEvents)
            self.event.callback = self._on_recognition
            
            logger.info("Windows语音识别引擎初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化Windows语音识别失败: {e}")
            return False
    
    def _on_recognition(self, event_type, stream_number, stream_position, recognition_type):
        """语音识别事件回调"""
        if event_type == 1:  # 识别成功
            try:
                result = self.event.GetResult()
                text = result.PhraseInfo.GetText()
                if text and self.callback:
                    self.callback(text)
            except Exception as e:
                logger.error(f"处理识别结果失败: {e}")
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始监听"""
        if not self.initialize():
            logger.error("无法启动Windows语音识别")
            self._start_fallback_mode(callback)
            return
        
        self.callback = callback
        self.is_listening = True
        
        logger.info("Windows语音识别已启动")
        print("请说中文...（说'停止监听'可以退出）")
        
        # 保持线程运行
        def keep_alive():
            while self.is_listening:
                time.sleep(0.1)
        
        self.recognizer_thread = threading.Thread(target=keep_alive, daemon=True)
        self.recognizer_thread.start()
    
    def _start_fallback_mode(self, callback: Callable[[str], None]):
        """降级模式：使用文本输入"""
        logger.info("使用文本输入模式")
        self.callback = callback
        self.is_listening = True
        
        def input_thread():
            while self.is_listening:
                try:
                    text = input("请输入文本（输入'停止监听'退出）: ").strip()
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
        try:
            if self.grammar:
                self.grammar.DictationSetState(0)  # 禁用听写
        except:
            pass
        logger.info("Windows语音识别已停止")


class SpeechRecognitionEvents:
    """语音识别事件处理类"""
    
    def __init__(self):
        self.callback = None
    
    def OnRecognition(self, stream_number, stream_position, recognition_type):
        """识别成功事件"""
        if self.callback:
            self.callback(1, stream_number, stream_position, recognition_type)
    
    def OnFalseRecognition(self, stream_number, stream_position, recognition_type):
        """错误识别事件"""
        logger.debug("语音识别错误或无法识别")
    
    def GetResult(self):
        """获取识别结果"""
        try:
            return self.recognizer_context.Recognizer().GetRecognizer().GetResult()
        except:
            return None


# 方法2: 使用Windows语音识别API（备用方案）
class WindowsSpeechAPI:
    """使用Windows语音识别API"""
    
    def __init__(self):
        self.is_listening = False
        self.callback = None
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始监听"""
        logger.info("启动Windows语音识别API")
        self.callback = callback
        self.is_listening = True
        
        # 由于直接调用Windows API较复杂，这里使用模拟模式
        self._start_simulation()
    
    def _start_simulation(self):
        """模拟Windows语音识别"""
        def simulation_thread():
            print("Windows语音识别模拟模式已启动")
            print("请说中文...（在实际应用中会调用真正的Windows语音识别）")
            
            while self.is_listening:
                try:
                    # 在实际应用中，这里会调用真正的Windows语音识别API
                    # 现在使用文本输入模拟
                    text = input("语音输入模拟（输入'停止监听'退出）: ").strip()
                    if not text:
                        continue
                    
                    if text.lower() in ['停止监听', '停止', 'quit', 'exit']:
                        self.stop_listening()
                        break
                    
                    if self.callback:
                        self.callback(text)
                        
                except Exception as e:
                    logger.error(f"模拟线程错误: {e}")
                    break
        
        threading.Thread(target=simulation_thread, daemon=True).start()
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        logger.info("Windows语音识别已停止")


# 方法3: 使用系统命令调用Windows语音识别
class SystemSpeechRecognizer:
    """通过系统命令调用语音识别"""
    
    def __init__(self):
        self.is_listening = False
        self.callback = None
    
    def start_listening(self, callback: Callable[[str], None]):
        """开始监听"""
        logger.info("启动系统语音识别")
        self.callback = callback
        self.is_listening = True
        
        # 使用PowerShell调用Windows语音识别
        self._start_powershell_listener()
    
    def _start_powershell_listener(self):
        """使用PowerShell监听"""
        def powershell_thread():
            import subprocess
            import re
            
            print("通过PowerShell启动语音识别...")
            
            # PowerShell命令来调用语音识别
            # 注意：这只是一个示例，实际需要更复杂的处理
            ps_command = """
            Add-Type -AssemblyName System.Speech
            $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
            $grammar = New-Object System.Speech.Recognition.DictationGrammar
            $recognizer.LoadGrammar($grammar)
            $recognizer.SetInputToDefaultAudioDevice()
            
            while ($true) {
                $result = $recognizer.Recognize()
                if ($result) {
                    Write-Output $result.Text
                }
                Start-Sleep -Milliseconds 100
            }
            """
            
            try:
                # 运行PowerShell命令
                process = subprocess.Popen(
                    ["powershell", "-Command", ps_command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                while self.is_listening and process.poll() is None:
                    line = process.stdout.readline()
                    if line and self.callback:
                        self.callback(line.strip())
                    
                    time.sleep(0.1)
                
                process.terminate()
                
            except Exception as e:
                logger.error(f"PowerShell语音识别失败: {e}")
                self._start_fallback_mode()
        
        threading.Thread(target=powershell_thread, daemon=True).start()
    
    def _start_fallback_mode(self):
        """降级到文本输入模式"""
        def input_thread():
            while self.is_listening:
                try:
                    text = input("语音识别不可用，请输入文本: ").strip()
                    if text and self.callback:
                        self.callback(text)
                except:
                    break
        
        threading.Thread(target=input_thread, daemon=True).start()
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False


# 工厂函数
def create_windows_speech_recognizer(method: str = "auto"):
    """
    创建Windows语音识别器
    method: "com" | "api" | "system" | "auto"
    """
    if method == "com":
        return WindowsSpeechRecognizer()
    elif method == "api":
        return WindowsSpeechAPI()
    elif method == "system":
        return SystemSpeechRecognizer()
    else:
        # 自动选择最佳方案
        try:
            return WindowsSpeechRecognizer()
        except:
            try:
                return WindowsSpeechAPI()
            except:
                return SystemSpeechRecognizer()


# 单例实例
speech_recognizer_instance = None

def get_speech_recognizer():
    """获取语音识别器实例"""
    global speech_recognizer_instance
    if speech_recognizer_instance is None:
        speech_recognizer_instance = create_windows_speech_recognizer()
    return speech_recognizer_instance


# 测试代码
if __name__ == "__main__":
    def test_callback(text):
        print(f"识别到: {text}")
    
    recognizer = get_speech_recognizer()
    print("开始测试Windows语音识别...")
    print("请说话（说'停止监听'可以退出）")
    
    recognizer.start_listening(test_callback)
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recognizer.stop_listening()
        print("测试结束")