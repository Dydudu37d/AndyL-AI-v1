import logging
import speech_recognition as sr

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("STT")

def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # 调整环境噪声
            logger.info("正在调整环境噪声，请保持安静...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("环境噪声调整完成")
            
            print("请开始说话...")
            audio = recognizer.listen(source, phrase_time_limit=15)
            print("录音完成")
        
        try:
            # 优先使用Google语音识别
            text = recognizer.recognize_google(audio, language="zh-CN")

            print(f"识别结果: {text}")
            print(f"你：{text}")
            return text
        except sr.RequestError as e:
            logger.warning(f"Google语音识别服务不可用: {e}")
            
            # 尝试使用Sphinx作为备选
            try:
                text = recognizer.recognize_sphinx(audio, language="zh-CN")
                print(f"Sphinx识别结果: {text}")
                print(f"你：{text}")
                return text
            except ImportError:
                logger.error("未安装pocketsphinx")
                return None
            except Exception as sphinx_error:
                logger.error(f"Sphinx识别失败: {sphinx_error}")
                return None
        
    except sr.UnknownValueError:
        print("无法识别音频")
        return None
    except sr.RequestError as e:
        print(f"请求错误: {e}")
        return None
    except Exception as e:
        logger.error(f"语音识别过程中发生错误: {e}")
        return None

if __name__ == "__main__":
    recognize_speech()
    