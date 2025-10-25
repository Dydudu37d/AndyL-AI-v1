import logging
import requests
import asyncio
from ai_brain import get_ai_brain
from tts_speaker import get_tts_speaker
from speech_recognizer import get_speech_recognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SimpleVTuberSystem:
    """简化版VTuber系统"""
    
    def __init__(self):
        self.ai_brain = get_ai_brain("deepseek-r1:1.5b")
        self.tts_speaker = get_tts_speaker()
        self.speech_recognizer = get_speech_recognizer()
    
    def check_ollama(self):
        """检查Ollama服务"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                print("✓ Ollama服务连接正常")
                return True
            else:
                print("✗ Ollama服务异常")
                return False
        except:
            print("✗ 无法连接到Ollama服务")
            print("请确保Ollama正在运行:")
            print("1. 检查任务管理器中是否有ollama.exe")
            print("2. 如果没有，请运行: ollama serve")
            print("3. 如果端口冲突，运行: taskkill /f /im ollama.exe 然后重新运行 ollama serve")
            return False
    
    async def run(self):
        """运行系统"""
        print("检查系统状态...")
        
        if not self.check_ollama():
            return
        
        # 初始化各组件
        try:
            # 初始化AI
            ai_ok = self.ai_brain.initialize(
                name="AndyL",
                personality="活泼可爱，有点调皮，喜欢开玩笑，对科技感兴趣",
                style="口语化，使用一些网络流行语和表情符号，但不过度"
            )
            if not ai_ok:
                return
            
            # 初始化TTS
            tts_ok = self.tts_speaker.initialize("zh-CN-XiaoxiaoNeural")
            if not tts_ok:
                return
            
            print("✓ 系统初始化完成")
            print("\n=== AI VTuber 交互模式 ===")
            print("输入 'quit' 退出程序")
            print("输入 'voice' 启动语音模式")
            print("输入 'text' 返回文本模式")
            
            # 问候
            await self.tts_speaker.speak_async("你好，我是小深，很高兴为你服务！")
            
            # 主循环
            while True:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'voice':
                    print("启动语音模式...")
                    self.speech_recognizer.start_listening(
                        lambda text: print(f"识别到: {text}")
                    )
                elif user_input.lower() == 'text':
                    print("返回文本模式")
                    self.speech_recognizer.stop_listening()
                else:
                    response = self.ai_brain.process_text(user_input, use_stream=False)
                    if response:
                        print(f"小深: {response}")
                        self.tts_speaker.speak(response)
                        
        except Exception as e:
            print(f"系统错误: {e}")

async def main():
    system = SimpleVTuberSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())