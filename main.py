import asyncio
import logging
import threading
import time
import os
import sys
from datetime import datetime

# 导入AI电脑控制系统
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai.ai_computer_controller import AIComputerController
from ctypes import windll
from interactive_obs_text_tool import OBSTextTool # textAI.modify_text_source("TextAI","你好，我是AndyL，很高兴和你聊天！")
from obs import obs_controller
from ai.ai_brain import AIBrain, VTuberPersona, get_ai_brain
from tts_speaker import get_tts_speaker,TextProcessor
import re_remove_words_example as re_remove
from windows_speech_recognizer import get_speech_recognizer as get_windows_speech_recognizer
from localai_speech_recognizer import get_speech_recognizer as get_unified_speech_recognizer
from stt_recognizer import get_speech_recognizer as get_stt_speech_recognizer
from mouse_controller import get_mouse_controller
from screen_capture import ScreenCapture, demo_screen_control
from dotenv import load_dotenv
from keyboard_shortcuts import start_keyboard_shortcuts, stop_keyboard_shortcuts, set_recording_callback
from pynput import keyboard
from ai.ai_thought_display import AIThoughtDisplay
from vtube_studio_controller import *
# 导入AI想法显示模块
from ai.ai_thought_display import get_ai_thought_display
# 导入语音训练模块
import chat_processing as chat
import twitch_subscription_handler as sub
# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

textAI = OBSTextTool()
obs_controller.set_current_scene("chat_room")
obs_controller.set_text_source_content("textai123", "")
obs_controller.set_text_source_content("textai", "")
textAI.text("textai","")

SetWindowPos = windll.user32.SetWindowPos

class VTuberSystem:
    """VTuber系统 - 整合AI、TTS、语音识别和Live2D形象控制"""
    
    def __init__(self, ai_type="ollama", ai_model=None, tts_type=None, stt_type=None, live2d_type=None, live2d_model_path=None):
        """
        初始化VTuber系统
        
        参数:
            ai_type: AI类型，可选值: "ollama", "localai"
            ai_model: AI模型名称，如果为None则使用默认模型
            tts_type: TTS类型，可选值: "windows", "localai"，如果为None则使用.env配置
            stt_type: STT类型，可选值: "windows", "localai", "stt"，如果为None则使用.env配置
            live2d_type: Live2D后端类型，可选值: "web", "direct", "vtube_studio"，如果为None则使用.env配置
        """
        # 根据AI类型选择模型
        if ai_type == "localai":
            model = ai_model or "gpt-3.5-turbo"
            self.ai_brain = get_ai_brain(model=model, ai_type="localai")
        else:
            # 默认使用Ollama
            model = ai_model or "llama3.2:latest"
            self.ai_brain = get_ai_brain(model=model)
        
        # 初始化TTS说话器
        self.tts_speaker = get_tts_speaker()
        
        # 确定TTS类型
        self.tts_type = tts_type or os.getenv("TTS_TYPE", "windows")
        self.textprocessor = TextProcessor()
        
        # 确定STT类型 - 默认使用stt.py的语音识别
        self.stt_type = stt_type or os.getenv("STT_TYPE", "stt")
        
        # 初始化语音识别器 - 使用统一的工厂函数，会根据stt_type选择对应的识别器
        self.speech_recognizer = get_unified_speech_recognizer(self.stt_type)
        
        # 初始化VTubeStudio控制器
        self.vts_controller = VTubeStudioController(port=8002)
        
        # 初始化Live2D控制器 (已廢棄)
        # self.live2d_type = live2d_type or os.getenv("LIVE2D_TYPE", "web")
        # self.live2d_model_path = live2d_model_path or os.getenv("LIVE2D_MODEL_PATH")
        # self.live2d_controller = get_live2d_controller(backend_type=self.live2d_type, model_path=self.live2d_model_path)
        
        # 初始化鼠标控制器
        self.mouse_controller = get_mouse_controller()
        
        # 初始化屏幕捕获器
        self.screen_capture = None
        self.is_screen_capturing = False
        
        self.is_initialized = False
        self.is_voice_mode = False
        self.ai_type = ai_type
        self.is_mouse_control_enabled = False
        self.ai_computer_controller = None  # AI电脑控制器
        self.is_ai_control_mode = False  # AI控制模式状态
        
        # 初始化键盘修饰键状态变量
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.shift_pressed = False
        
        # 初始化语音训练器
        self.win_pressed = False  # 添加Win键状态变量
        self.shortcut_detected = False
        
        # 初始化监听器变量
        self.listener = None
        
        # 初始化AI想法显示管理器
        self.ai_thought_display = get_ai_thought_display(textAI)
    
    def initialize(self) -> bool:
        """初始化系统"""
        # 初始化AI电脑控制器
        self.ai_computer_controller = AIComputerController()
        
        # 根据AI类型配置AI电脑控制器
        if self.ai_type == "localai":
            # 使用LocalAI
            localai_host = os.getenv("LOCALAI_HOST", "localhost")
            localai_port = int(os.getenv("LOCALAI_PORT", "8080"))
            localai_model = os.getenv("LOCALAI_MODEL", "gpt-3.5-turbo")
            self.ai_computer_controller.set_localai_config(localai_host, localai_port, localai_model)
            logger.info(f"LocalAI配置已设置: {localai_host}:{localai_port}, 模型={localai_model}")
        elif self.ai_type == "ollama":
            # 使用Ollama
            ollama_host = os.getenv("OLLAMA_HOST", "localhost")
            ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
            self.ai_computer_controller.set_ollama_config(ollama_host, ollama_port, ollama_model)
            logger.info(f"Ollama配置已设置: {ollama_host}:{ollama_port}, 模型={ollama_model}")
        else:
            # 使用API类型
            ai_api_url = os.getenv("AI_API_URL", "")
            ai_api_key = os.getenv("AI_API_KEY", "")
            
            if ai_api_url and ai_api_key:
                self.ai_computer_controller.set_ai_api_config(ai_api_url, ai_api_key)
                logger.info("AI API配置已从环境变量加载")
            else:
                logger.warning("未设置AI API配置，请在.env文件中设置AI_API_URL和AI_API_KEY")
        # 初始化AI大脑
        ai_success = self.ai_brain.initialize(
            name="AndyL",
            personality="活泼可爱，有点调皮，喜欢开玩笑，对科技感兴趣，讨厌句子里面夹英文，喜欢要麼全中文，要麼全英文，不喜欢句子里面带表情包",
            style="口语化，使用一些网络流行语和表情符号，但不过度，讨厌句子里面夹英文，喜欢要麼全中文，要麼全英文，不喜欢句子里面带表情包"
        )
        
        if not ai_success:
            logger.error("AI大脑初始化失败")
            return False
        
        # 设置TTS类型
        if hasattr(self.tts_speaker, 'set_tts_type'):
            self.tts_speaker.set_tts_type(self.tts_type)
        
        # 初始化TTS说话器 - 使用同步初始化
        tts_success = self.tts_speaker.initialize("zh-CN-XiaoxiaoNeural")
        
        if not tts_success:
            logger.error("TTS说话器初始化失败，使用降级模式")
            # 继续运行，只是TTS可能不可用
        
        # 初始化键盘快捷键
        set_recording_callback(self.toggle_voice_recording)
        
        # 初始化鼠标控制器
        mouse_success = self.mouse_controller.initialize()
        if not mouse_success:
            logger.warning("鼠标控制器初始化失败，功能可能受限")
        
        # 初始化屏幕捕获器
        self.screen_capture = ScreenCapture(capture_interval=0.7)
        self.screen_capture.set_mouse_controller(self.mouse_controller)
        self.screen_capture.set_ai_brain(self.ai_brain)
        
        self.is_voice_mode = False
        self.is_initialized = True
        logger.info(f"VTuber系统初始化完成 (使用{self.ai_type}, TTS: {self.tts_type}, STT: {self.stt_type}, VTubeStudio: {self.vts_controller.ws_ip}:{self.vts_controller.port})")
        
        return True
    
    def process_input(self, user_input: str):
        """处理用户输入"""
        if not self.is_initialized:
            logger.error("系统未初始化")
            return
        
        logger.info(f"用户输入: {user_input}")
        
        # 检查是否是控制命令
        if self._process_control_command(user_input):
            return
        
        # 如果正在说话，先停止
        if self.tts_speaker.is_speaking_now():
            self.tts_speaker.stop()
        
        # 显示AI正在思考
        #self.ai_thought_display.display_thought(f"正在思考用户的问题：{user_input[:30]}...")
        
        # 使用流式处理获取AI响应
        try:
            # 首先检查vts_controller是否存在
            if not hasattr(self, 'vts_controller') or self.vts_controller is None:
                logger.warning("VTS控制器未初始化")
                self.vts_expressions = []
            else:
                # 检查连接状态，确保在获取表情列表前已连接
                try:
                    # 尝试获取当前事件循环并运行异步任务
                    loop = asyncio.get_event_loop()
                    # 如果当前线程没有事件循环，则创建一个
                    if loop.is_running():
                        # 在单独的线程中运行异步任务
                        import threading
                        result = [None]
                        def run_in_new_loop():
                            try:
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                
                                # 检查并尝试连接VTube Studio
                                if not hasattr(self.vts_controller, 'connected') or not self.vts_controller.connected:
                                    logger.info("尝试连接VTube Studio...")
                                    connected = new_loop.run_until_complete(self.vts_controller.connect())
                                    if not connected:
                                        logger.warning("无法连接到VTube Studio")
                                        result[0] = []
                                        return
                                
                                result[0] = new_loop.run_until_complete(self.vts_controller.get_expressions())
                            except Exception as e:
                                logger.error(f"在新线程中获取表情列表失败: {e}")
                                result[0] = []
                            finally:
                                if 'new_loop' in locals():
                                    new_loop.close()
                        thread = threading.Thread(target=run_in_new_loop)
                        thread.start()
                        thread.join()
                        self.vts_expressions = result[0] if result[0] is not None else []
                    else:
                        # 在当前事件循环中运行异步任务
                        # 检查并尝试连接VTube Studio
                        if not hasattr(self.vts_controller, 'connected') or not self.vts_controller.connected:
                            logger.info("尝试连接VTube Studio...")
                            connected = loop.run_until_complete(self.vts_controller.connect())
                            if not connected:
                                logger.warning("无法连接到VTube Studio")
                                self.vts_expressions = []
                            else:
                                self.vts_expressions = loop.run_until_complete(self.vts_controller.get_expressions())
                        else:
                            self.vts_expressions = loop.run_until_complete(self.vts_controller.get_expressions())
                except Exception as e:
                    logger.error(f"检查连接状态时出错: {e}")
                    self.vts_expressions = []
        except RuntimeError:
            # 如果无法获取事件循环，则使用asyncio.run
            try:
                self.vts_expressions = asyncio.run(self.vts_controller.get_expressions())
            except RuntimeError:
                # 如果仍然失败，则使用空的表情列表
                self.vts_expressions = []
        except Exception as e:
            # 捕获所有其他异常
            logger.error(f"获取表情列表时发生未知错误: {e}")
            self.vts_expressions = []
        response_generator = self.ai_brain.process_text(f"{user_input}\n你可以使用的表情列表: {[exp.get('expressionName', '') if isinstance(exp, dict) else str(exp) for exp in self.vts_expressions]}\n请根据用户输入和可用表情列表进行回复，并严格按照以下格式输出：\n[你的回复内容, [表情名称,是否激活(true/false),淡入淡出时间(秒)]]\n注意事项：\n1. 回复内容应自然流畅，符合对话情境\n2. 表情名称必须从提供的表情列表中选择，不要使用列表外的表情\n3. 是否激活使用true或false表示，淡入淡出时间使用数字（如0.3）表示\n4. 不需要在回复中提及当前表情列表或格式要求\n5. 确保输出格式严格正确，不要添加额外内容", use_stream=True)
        
        # 收集完整响应
        full_response = ""
        plain_text_response = ""
        print("AndyL: ", end="", flush=True)
        
        # 记录生成的响应片段，用于显示部分想法
        thinking_chunks = []
        content_buffer = ""
        
        for chunk in response_generator:
            full_response += chunk
            content_buffer += chunk
            
            # 检查是否有完整的响应结构
            if '[' in content_buffer and ']' in content_buffer:
                try:
                    # 尝试提取纯文本内容（兼容不同格式）
                    # 首先尝试标准格式：[回复内容, [表情参数]]
                    if content_buffer.count('[') >= 2 and content_buffer.count(']') >= 2:
                        # 找到第一个[和第二个[的位置
                        first_bracket = content_buffer.index('[')
                        second_bracket_pos = content_buffer.find('[', first_bracket + 1)
                        
                        if second_bracket_pos != -1:
                            # 提取回复内容
                            response_content = content_buffer[first_bracket + 1:second_bracket_pos - 1].strip()
                            
                            # 如果有新的回复内容，打印出来
                            if response_content and response_content not in ''.join(thinking_chunks):
                                print(response_content, end="", flush=True)
                                thinking_chunks.append(response_content)
                                plain_text_response = response_content
                    # 如果标准格式解析失败，尝试直接提取文本内容
                        elif not plain_text_response:
                            # 去掉可能的括号和格式标记
                            text_only = content_buffer.replace('[', '').replace(']', '').strip()
                            filtered_response = re_remove.remove_json_tokens(text_only)
                            # 由于text_only已经去掉了括号，这里直接使用处理后的文本即可
                            filtered_response = self.textprocessor._filter_text(filtered_response)
                            if filtered_response and filtered_response not in ''.join(thinking_chunks):
                                # 排除纯数字和特殊字符的情况
                                if any(char.isalpha() for char in filtered_response):
                                    print(filtered_response, end="", flush=True)
                                    thinking_chunks.append(filtered_response)
                                    plain_text_response = filtered_response
                    obs_controller.set_text_source_content("textai", filtered_response)
                    obs_controller.set_text_source_content("textai123", filtered_response)
                    textAI.text("textai",filtered_response)
                                
                except Exception as e:
                    logger.warning(f"解析响应内容时出错: {e}")
        
        # 确保即使解析失败也有内容显示
        if not plain_text_response and full_response:
            # 作为最后的备用方案，直接显示完整响应中的文本部分
            text_only = full_response.replace('[', '').replace(']', '').strip()
            if text_only and text_only not in ''.join(thinking_chunks):
                # 排除纯数字和特殊字符的情况
                if any(char.isalpha() for char in text_only):
                    print(text_only, end="", flush=True)
                    plain_text_response = text_only
        
        print("\n")
        
        # 显示最终的完整想法（可以选择在几秒后清除）
        # self.ai_thought_display.display_thought(f"{full_response[:80]}...", clear_after=10)
        
        # 设置表情（使用与获取表情列表相同的健壮方法）
        try:
            # 尝试解析响应中的表情参数（更健壮的方式）
            expression_name = ""
            is_active = False
            fade_time = 0.3
            
            if '[' in full_response and ']' in full_response and full_response.count('[') >= 2:
                # 找到第一个[和最后一个]的位置，提取整个响应内容
                first_bracket = full_response.index('[')
                last_bracket = full_response.rindex(']')
                response_content = full_response[first_bracket + 1:last_bracket].strip()
                
                # 尝试找到第二个[的位置，分离回复内容和表情参数
                if '[' in response_content:
                    second_bracket = response_content.index('[')
                    # 提取表情参数部分
                    expression_params_str = response_content[second_bracket:].strip()
                    
                    # 尝试解析表情参数
                    if expression_params_str.startswith('[') and expression_params_str.endswith(']'):
                        # 去除外层括号并分割参数
                        params = expression_params_str[1:-1].split(',')
                        if len(params) >= 3:
                            expression_name = params[0].strip().strip('"\'')
                            # 确保第二个参数是布尔值
                            is_active = params[1].strip().lower() == "true" or params[1].strip() == "1"
                            fade_time = float(params[2].strip()) if len(params) > 2 and params[2].strip() else 0.3
            
            # 只有当表情名称不为空时才设置表情
                if expression_name:
                    # 使用与获取表情列表相同的健壮异步处理方式
                    try:
                        # 检查当前事件循环状态
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 在单独的线程中运行异步任务
                            def run_expression_in_new_loop():
                                try:
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    new_loop.run_until_complete(self.vts_controller.set_expression(expression_name, is_active, fade_time))
                                except Exception as e:
                                    logger.warning(f"在新线程中设置表情失败: {e}")
                                finally:
                                    # 安全关闭新的事件循环
                                    try:
                                        # 检查事件循环是否正在运行
                                        if new_loop.is_running():
                                            # 取消所有任务
                                            for task in asyncio.all_tasks(new_loop):
                                                task.cancel()
                                            # 等待所有任务取消
                                            new_loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(new_loop), return_exceptions=True))
                                    except:
                                        pass
                                    try:
                                        new_loop.close()
                                    except:
                                        pass
                            # 创建并启动线程
                            thread = threading.Thread(target=run_expression_in_new_loop)
                            thread.daemon = True
                            thread.start()
                            # 不等待线程完成，避免阻塞
                        else:
                            # 在当前事件循环中运行异步任务
                            try:
                                loop.run_until_complete(self.vts_controller.set_expression(expression_name, is_active, fade_time))
                            except RuntimeError as e:
                                # 处理事件循环已关闭的情况
                                if "Event loop is closed" in str(e):
                                    logger.warning("当前事件循环已关闭，创建新的事件循环来设置表情")
                                    # 创建新的事件循环
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    try:
                                        new_loop.run_until_complete(self.vts_controller.set_expression(expression_name, is_active, fade_time))
                                    finally:
                                        new_loop.close()
                                else:
                                    raise
                    except Exception as e:
                        logger.error(f"设置表情时出错: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"解析表情参数失败: {e}")
        except Exception as e:
            logger.error(f"设置表情时出错: {e}")
        
        # 使用TTS播放响应
        if plain_text_response and not plain_text_response.startswith("错误"):
            # 播放纯文本语音
            # 优化文本处理：移除不需要的JSON标记
            cleaned_text = re_remove.remove_json_tokens(plain_text_response)
            # 确保文本格式正确
            cleaned_text = self.textprocessor._filter_text(cleaned_text)
            self.tts_speaker.speak(cleaned_text)
        elif full_response and not full_response.startswith("错误"):
            # 备用方案：如果无法提取纯文本，则使用完整响应（但会过滤掉表情参数部分）
            try:
                if '[' in full_response and ']' in full_response and full_response.count('[') >= 2:
                    first_bracket = full_response.index('[')
                    second_bracket = full_response.index('[', first_bracket + 1)
                    filtered_response = full_response[first_bracket + 1:second_bracket - 1].strip()
                    filtered_response = re_remove.remove_json_tokens(filtered_response)
                    # 直接使用处理后的文本，不再调用extract_plain_text_from_response
                    filtered_response = self.textprocessor._filter_text(filtered_response)
                    if filtered_response:
                        self.tts_speaker.speak(filtered_response)
            except:
                pass
    
    def _process_control_command(self, command: str) -> bool:
        """
        处理控制命令
        
        返回:
            如果是控制命令并已处理，则返回True
        """
        command_lower = command.lower()
        
        # 屏幕捕获控制命令
        if "开始截图" in command_lower or "开始屏幕捕获" in command_lower:
            if not self.is_screen_capturing and self.screen_capture:
                self.screen_capture.start_capturing()
                self.is_screen_capturing = True
                print("已开始定时屏幕捕获（每0.7秒一次）")
                # self.tts_speaker.speak("已开始定时屏幕捕获，每零点七秒一次")
                return True
            elif self.is_screen_capturing:
                print("屏幕捕获已在运行中")
                # self.tts_speaker.speak("屏幕捕获已在运行中")
                return True
        elif "停止截图" in command_lower or "停止屏幕捕获" in command_lower:
            if self.is_screen_capturing and self.screen_capture:
                self.screen_capture.stop_capturing()
                self.is_screen_capturing = False
                print("已停止定时屏幕捕获")
                # self.tts_speaker.speak("已停止定时屏幕捕获")
                return True
            elif not self.is_screen_capturing:
                print("屏幕捕获未在运行中")
                # self.tts_speaker.speak("屏幕捕获未在运行中")
                return True
        elif "截图" in command_lower or "屏幕截图" in command_lower:
            if self.screen_capture:
                self.screen_capture.set_ai_brain(self.ai_brain)
                screenshot = self.screen_capture.capture_screen()        
                screen_capture_path = self.screen_capture.save_screenshot(screenshot=screenshot)
                if screen_capture_path:
                    print(f"已保存当前屏幕截图到: {screen_capture_path}")
                    # self.tts_speaker.speak("已保存当前屏幕截图")
                if screenshot is not None:
                    print("已捕获当前屏幕")
                    # self.tts_speaker.speak("已捕获当前屏幕")
                    # 可选：显示截图
                    if "显示" in command_lower:
                        self.screen_capture.show_captured_screen()
        elif "評價" in command_lower or "評價如何" in command_lower:
            if self.screen_capture:
                screenshot = self.screen_capture.capture_screen()
                # 创建一个临时文件名来保存截图
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screen_capture_path = f"screenshots/screenshot_{timestamp}.png"
                
                # 保存截图
                save_success = self.screen_capture.save_screenshot(file_path=screen_capture_path, screenshot=screenshot)
                if save_success:
                    print(f"已保存当前屏幕截图到: {screen_capture_path}")
                    # self.tts_speaker.speak("已保存当前屏幕截图")
                if screenshot is not None:
                    print("已捕获当前屏幕")
                    # self.tts_speaker.speak("已捕获当前屏幕")
                    # 可选：显示截图
                    if save_success:
                        # 创建AIComputerController实例
                        ai_controller = AIComputerController()
                        # 调用_send_to_ollama获取AI对截图的评价，设置need_commands=False直接获取文本响应
                        evaluation_result = ai_controller._send_to_ollama(
                            screenshot_path=screen_capture_path,
                            prompt="評價/形容一下我的桌面截图(说出你的观点就行)\n",
                            need_commands=False
                        )
                        if evaluation_result:
                            # 使用TTS朗读评价结果
                            self.tts_speaker.speak(evaluation_result)

            return True
        
        # 鼠标控制命令
        if "移动鼠标" in command_lower:
            try:
                # 尝试解析坐标，如"移动鼠标到100,200"
                parts = command_lower.split("到")
                if len(parts) > 1:
                    coords = parts[1].strip().split(",")
                    if len(coords) == 2:
                        x = int(coords[0])
                        y = int(coords[1])
                        self.mouse_controller.move_mouse(x, y)
                        print(f"已移动鼠标到: {x}, {y}")
                        # self.tts_speaker.speak(f"已将鼠标移动到坐标{x}逗号{y}")
                        return True
            except Exception as e:
                logger.error(f"处理鼠标移动命令失败: {e}")
        elif "点击鼠标" in command_lower:
            button = "left"
            if "右键" in command_lower:
                button = "right"
            elif "中键" in command_lower:
                button = "middle"
            
            count = 1
            if "双击" in command_lower:
                count = 2
            
            self.mouse_controller.click(button=button, count=count)
            action = "双击" if count == 2 else "点击"
            btn_name = "右键" if button == "right" else "中键" if button == "middle" else "左键"
            print(f"已{action}鼠标{btn_name}")
            # self.tts_speaker.speak(f"已执行鼠标{btn_name}{action}")
            return True
        elif "滚动" in command_lower:
            try:
                # 尝试解析滚动量，如"向下滚动5次"
                direction = 1  # 向上
                if "向下" in command_lower:
                    direction = -1
                
                amount = 1
                # 简单的数字提取
                for word in command_lower.split():
                    if word.isdigit():
                        amount = int(word)
                        break
                
                self.mouse_controller.scroll(0, direction * amount)
                dir_text = "向下" if direction == -1 else "向上"
                print(f"已{dir_text}滚动{amount}次")
                # self.tts_speaker.speak(f"已{dir_text}滚动鼠标{amount}次")
                return True
            except Exception as e:
                logger.error(f"处理滚动命令失败: {e}")
        # 键盘控制命令
        elif "输入" in command_lower or "打字" in command_lower:
            try:
                # 尝试提取要输入的文本，如"输入你好"
                if "输入" in command_lower:
                    text = command_lower.split("输入")[1].strip()
                else:
                    text = command_lower.split("打字")[1].strip()
                
                if text:
                    self.mouse_controller.type_string(text)
                    print(f"已输入文本: {text}")
                    # self.tts_speaker.speak(f"已输入文本{text}")
                    return True
            except Exception as e:
                logger.error(f"处理输入命令失败: {e}")
        # 控制模式切换
        elif "启用鼠标控制" in command_lower:
            self.is_mouse_control_enabled = True
            print("鼠标控制功能已启用")
            # self.tts_speaker.speak("鼠标控制功能已启用")
            return True
        elif "禁用鼠标控制" in command_lower:
            self.is_mouse_control_enabled = False
            print("鼠标控制功能已禁用")
            # self.tts_speaker.speak("鼠标控制功能已禁用")
            return True
        
        # 如果启用了AI控制模式，让AI来解析复杂命令
        if self.is_mouse_control_enabled:
            return self._process_ai_control_command(command)
        
        return False
    
    def _process_ai_control_command(self, command: str) -> bool:
        """处理AI控制命令"""
        try:
            # 让AI解析命令并生成控制指令
            system_prompt = f"请将用户的命令转换为电脑控制指令: {command}\n"
            system_prompt += "请分析屏幕截图和用户提示，然后返回一系列控制指令。"
            system_prompt += "指令格式必须是用|分隔的命令列表，例如: 'm_up|m_left|m_click|t_hello'"
            system_prompt += "请只返回命令列表，不要包含任何额外的解释文本。"
            system_prompt += "\n\n可用命令:\n"
            system_prompt += "- move_mouse|x|y 移动鼠标\n"
            system_prompt += "- m_click/m_right_click/m_double_click: 鼠标点击\n"
            system_prompt += "- type_string|text|[Text] ：输入文本，例如type_string|text|hello 将输入hello\n"
            system_prompt += "重要提示：参数必须是具体的数值，不能是'x'或'y'这样的占位符！\n"
            control_response = self.ai_brain.process_text(system_prompt)
            
            if not control_response or "错误" in control_response:
                return False
            
            # 解析AI返回的控制指令
            control_parts = control_response.strip().split("|")
            if len(control_parts) < 2:
                return False
            
            
            action = control_parts[0]
            params = control_parts[1:]
            
            # 执行控制指令
            if action == "move_mouse" and len(params) >= 2:
                # 添加更健壮的坐标解析
                try:
                    # 检查参数是否为实际数值而不是占位符
                    if params[0].lower() == 'x' or params[1].lower() == 'y':
                        # 如果是占位符，根据命令内容提供合理的默认值
                        if "右上" in command or "右上角" in command:
                            x, y = 1800, 100  # 右上角坐标（假设屏幕分辨率为1920x1080）
                        elif "右下" in command or "右下角" in command:
                            x, y = 1800, 980  # 右下角坐标
                        elif "左上" in command or "左上角" in command:
                            x, y = 100, 100   # 左上角坐标
                        elif "左下" in command or "左下角" in command:
                            x, y = 100, 980   # 左下角坐标
                        elif "中央" in command or "中心" in command:
                            x, y = 960, 540   # 中心坐标
                        else:
                            x, y = 960, 540   # 默认坐标
                        logger.warning(f"使用默认坐标代替占位符参数: {x}, {y}")
                    else:
                        x, y = int(params[0]), int(params[1])
                    
                    self.mouse_controller.move_mouse(x, y)
                    # self.tts_speaker.speak(f"已将鼠标移动到指定位置")
                    return True
                except ValueError as e:
                    logger.error(f"解析鼠标坐标失败: {e}")
                    # 尝试使用当前位置附近的坐标
                    current_x, current_y = self.mouse_controller.get_mouse_position()
                    try:
                        # 尝试从命令中提取数值作为偏移量
                        numbers = [int(word) for word in command.split() if word.isdigit()]
                        if numbers:
                            x, y = current_x + numbers[0], current_y + (numbers[1] if len(numbers) > 1 else 0)
                        else:
                            # 如果没有数值，使用默认偏移量
                            x, y = current_x + 100, current_y
                        self.mouse_controller.move_mouse(x, y)
                        # self.tts_speaker.speak(f"已尝试移动鼠标")
                        return True
                    except:
                        return False
                         
            elif action == "click" and len(params) >= 2:
                button = params[1] if len(params) > 1 else "left"
                count = 1
                if len(params) > 2:
                    try:
                        count = int(params[2])
                    except ValueError:
                        count = 1
                self.mouse_controller.click(button=button, count=count)
                # self.tts_speaker.speak(f"已执行鼠标点击")
                return True
            elif action == "scroll" and len(params) >= 2:
                try:
                    dx, dy = int(params[0]), int(params[1])
                    self.mouse_controller.scroll(dx, dy)
                    # self.tts_speaker.speak(f"已执行鼠标滚动")
                    return True
                except ValueError:
                    return False
            elif action == "type_string" and len(params) >= 1:
                text = params[0]
                self.mouse_controller.type_string(text)
                # self.tts_speaker.speak(f"已输入文本")
                return True
            elif action == "press_key" and len(params) >= 1:
                key = params[0]
                self.mouse_controller.press_key(key)
                return True
            elif action == "release_key" and len(params) >= 1:
                key = params[0]
                self.mouse_controller.release_key(key)
                return True
            elif action == "press_combination" and len(params) >= 1:
                keys = params
                self.mouse_controller.press_combination(keys)
                return True
        except Exception as e:
            logger.error(f"处理AI控制命令失败: {e}")
            # 尝试使用备用解析方法
            try:
                # 直接处理常见命令
                command_lower = command.lower()
                if "右上" in command_lower or "右上角" in command_lower:
                    self.mouse_controller.move_mouse(1800, 100)
                    # self.tts_speaker.speak(f"已将鼠标移动到右上角")
                    return True
                elif "右下" in command_lower or "右下角" in command_lower:
                    self.mouse_controller.move_mouse(1800, 980)
                    # self.tts_speaker.speak(f"已将鼠标移动到右下角")
                    return True
                elif "左上" in command_lower or "左上角" in command_lower:
                    self.mouse_controller.move_mouse(100, 100)
                    # self.tts_speaker.speak(f"已将鼠标移动到左上角")
                    return True
                elif "左下" in command_lower or "左下角" in command_lower:
                    self.mouse_controller.move_mouse(100, 980)
                    # self.tts_speaker.speak(f"已将鼠标移动到左下角")
                    return True
                elif "中央" in command_lower or "中心" in command_lower:
                    self.mouse_controller.move_mouse(960, 540)
                    # self.tts_speaker.speak(f"已将鼠标移动到屏幕中心")
                    return True
            except:
                pass
        
        return False
    
    def voice_recognition_callback(self, text: str):
        """语音识别回调函数"""
        print(f"[调试] 语音识别回调接收到文本: '{text}'")
        logger.info(f"语音识别到: {text}")
        # 直接打印"你："前缀的识别结果
        print(f"你：{text}")
        
        # 检查特殊命令
        if any(cmd in text.lower() for cmd in ["退出语音", "停止语音", "关闭语音", "stop"]):
            print("检测到退出语音命令")
            self.stop_voice_mode()
            return
        
        # 在新线程中处理语音输入
        def process_voice_input():
            try:
                # 显示AI正在思考
                # self.ai_thought_display.display_thought(f"正在理解语音输入：{text[:30]}...")
                
                # 使用流式处理获取AI响应
                response_generator = self.ai_brain.process_text(text, use_stream=False)
                
                # 收集完整响应
                full_response = ""
                print("AndyL: ", end="", flush=True)
                
                # 记录生成的响应片段，用于显示部分想法
                thinking_chunks = []
                
                for chunk in response_generator:
                    full_response += chunk
                    thinking_chunks.append(chunk)
                    
                    # 每积累一些内容就更新一次想法显示
                    if len(''.join(thinking_chunks)) > 20 or chunk.endswith((",", ".", "!", "?")):
                        partial_thought = ''.join(thinking_chunks)
                        # self.ai_thought_display.display_thought(f"正在组织语言... {partial_thought[:50]}...")
                        thinking_chunks = []
                    
                    print(chunk, end="", flush=True)
                
                print("\n")
                
                # 显示最终的完整想法（可以选择在几秒后清除）
                # self.ai_thought_display.display_thought(f"{full_response[:80]}...", clear_after=10)
                
                if full_response and not full_response.startswith("错误"):
                    # 根据响应文本自动应用Live2D表情和动作
                    # 播放语音
                    filtered_response = re_remove.extract_plain_text_from_response(f"[{full_response}]")
                    filtered_response = self.textprocessor._filter_text(filtered_response)
                    self.tts_speaker.speak(filtered_response)
            except Exception as e:
                logger.error(f"处理语音输入失败: {e}")
        
        processing_thread = threading.Thread(target=process_voice_input, daemon=True)
        processing_thread.start()
    
    def start_voice_mode(self):
        """启动语音模式"""
        if not self.is_initialized:
            logger.error("系统未初始化")
            return
        
        if self.is_voice_mode:
            logger.warning("已经在语音模式中")
            return
        
        self.is_voice_mode = True
        
        # 兼容不同的语音识别器接口
        if hasattr(self.speech_recognizer, 'start_realtime_listening'):
            self.speech_recognizer.start_realtime_listening(self.voice_recognition_callback)
        elif hasattr(self.speech_recognizer, 'start_listening'):
            self.speech_recognizer.start_listening(self.voice_recognition_callback)
        
        print("语音模式已启动，请开始说话...")
        print("说'退出语音'可以退出语音模式")
    
    def stop_voice_mode(self):
        """停止语音模式"""
        if not self.is_voice_mode:
            return
        
        self.is_voice_mode = False
        self.speech_recognizer.stop_listening()
        print("语音模式已停止")
    
    def toggle_voice_recording(self):
        """切换语音录音状态"""
        self.start_voice_mode()
        self.is_voice_mode = not self.is_voice_mode
    
    def toggle_voice_mode(self):
        """切换语音模式"""
        self.toggle_voice_recording()
    
    def start_ai_control_mode(self, custom_prompt=None):
        """启动AI控制模式（持续运行直到用户停止，可自定义控制行为）"""
        if not self.is_initialized:
            logger.error("系统未初始化")
            return
        
        if self.is_ai_control_mode:
            logger.warning("已经在AI控制模式中")
            # self.tts_speaker.speak("已经在AI控制模式中")
            return
        
        self.is_ai_control_mode = True
        
        def ai_control_loop():
            try:
                print("AI控制模式已启动，开始持续执行控制循环...")
                # self.tts_speaker.speak("AI控制模式已启动，开始持续执行控制循环")
                
                cycle_count = 0
                success_count = 0
                
                # 控制行为类型列表，用于轮换不同的控制操作
                control_behaviors = [
                    "AI自动探索桌面环境",
                    "AI进行随机鼠标移动测试",
                    "AI模拟用户操作演示",
                    "AI执行混合控制指令测试",
                    "AI进行多样化电脑控制"
                ]
                
                # 持续运行控制循环，直到用户停止
                while self.is_ai_control_mode:
                    try:
                        # 选择当前的提示信息：如果用户提供了自定义提示，则使用自定义提示；否则轮换使用预设的控制行为
                        current_prompt = custom_prompt if custom_prompt else control_behaviors[cycle_count % len(control_behaviors)]
                        
                        # 运行单个控制循环，使用轮换的提示信息
                        result = self.ai_computer_controller.run_single_cycle(
                            prompt=current_prompt, 
                            use_mock=False  # 改为使用真实的AI响应
                        )
                        
                        cycle_count += 1
                        if result["success"]:
                            success_count += 1
                            logger.info(f"控制循环 {cycle_count} 执行成功 (行为: {current_prompt})")
                        else:
                            logger.warning(f"控制循环 {cycle_count} 执行失败 (行为: {current_prompt})")
                    except Exception as e:
                        logger.error(f"单个控制循环执行失败: {e}")
                        cycle_count += 1
                    
                    # 每隔几个循环报告一次状态
                    if cycle_count % 5 == 0:
                        status_msg = f"已执行{cycle_count}个控制循环，成功{success_count}个"
                        logger.info(status_msg)
                        # 即使在非语音模式下也提供清晰的控制台反馈
                        print(f"[AI控制状态] {status_msg}")
                        # self.tts_speaker.speak(status_msg)
                    
                    # 控制循环间隔2秒
                    import time
                    time.sleep(2.0)
                
                # 统计最终结果
                print(f"\n===== AI控制模式已停止 =====")
                print(f"总共执行 {cycle_count} 个控制循环")
                print(f"成功 {success_count} 个循环")
                print(f"失败 {cycle_count - success_count} 个循环")
                success_rate = (success_count / cycle_count * 100) if cycle_count > 0 else 0
                print(f"成功率: {success_rate:.1f}%")
                print(f"==========================\n")
                # self.tts_speaker.speak(f"AI控制模式已停止，总共执行了{cycle_count}个循环，成功{success_count}个")
            except Exception as e:
                logger.error(f"AI控制循环执行异常: {e}")
                # self.tts_speaker.speak("AI控制循环发生异常")
            finally:
                # 确保状态被正确重置
                self.is_ai_control_mode = False
        
        # 在新线程中运行AI控制循环
        ai_control_thread = threading.Thread(target=ai_control_loop, daemon=True)
        ai_control_thread.start()
    
    def stop_ai_control_mode(self):
        """停止AI控制模式"""
        if not self.is_ai_control_mode:
            return
        
        self.is_ai_control_mode = False
        print("AI控制模式已停止")
        # self.tts_speaker.speak("AI控制模式已停止")
    
    def shutdown(self):
        """关闭系统"""
        self.stop_voice_mode()
        self.stop_ai_control_mode()
        stop_keyboard_shortcuts()
        if hasattr(self, 'mouse_controller'):
            self.mouse_controller.shutdown()
        if hasattr(self, 'ai_computer_controller'):
            self.ai_computer_controller.shutdown()
        
        sys.exit(0)
    
    def on_press(self, key):
        """键盘按下事件处理"""
        try:
            # 记录所有按键
            key_str = str(key)
            #   logger.debug(f"按键检测到: {key_str}")
            #   print(f"按键检测到: {key_str}")
            
            # 检测Ctrl键
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = True
                #logger.debug("Ctrl键已按下")
                # print("✅ Ctrl键状态: 已按下")
            # 检测Alt键
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_pressed = True
                #logger.debug("Alt键已按下")
                # print("✅ Alt键状态: 已按下")
            # 检测Shift键(额外检测，用于调试)
            elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.shift_pressed = True
                #logger.debug("Shift键已按下")
                # print("✅ Shift键状态: 已按下")
            # 检测Win键
            elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_r:
                self.win_pressed = True
                #logger.debug("Win键已按下")
                # print("✅ Win键状态: 已按下")
            # 检测R键
            elif (hasattr(key, 'char') and key.char and key.char.lower() == 'r') or str(key) == '<82>' or str(key) == '\x12' or str(key) == 'r':
                #logger.debug("检测到R键")
                # print("✅ R键已按下")
                # 检查是否同时按下了Ctrl和Alt
                if self.ctrl_pressed and self.alt_pressed:
                    #logger.info("✅ 成功检测到Ctrl+Alt+R组合键!")
                    # print("✅ 成功检测到Ctrl+Alt+R组合键!")
                    self.shortcut_detected = True
                    # 打印成功信息到控制台
                    self.toggle_voice_mode()
            # 打印当前修饰键状态
            # logger.debug(f"当前修饰键状态 - Ctrl: {self.ctrl_pressed}, Alt: {self.alt_pressed}, Shift: {self.shift_pressed}")
            # print(f"当前修饰键状态 - Ctrl: {self.ctrl_pressed}, Alt: {self.alt_pressed}, Shift: {self.shift_pressed}")
        except Exception as e:
            logger.error(f"处理按键按下事件时出错: {e}")
            # print(f"❌ 错误: {e}")
    
    def on_release(self, key):
        """键盘释放事件处理"""
        try:
            # 检测Ctrl键释放
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = False
                #logger.debug("Ctrl键已释放")
                #print("✅ Ctrl键状态: 已释放")
            # 检测Alt键释放
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_pressed = False
                #logger.debug("Alt键已释放")
                #print("✅ Alt键状态: 已释放")
            # 检测Shift键释放
            elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.shift_pressed = False
                #logger.debug("Shift键已释放")
                #print("✅ Shift键状态: 已释放")
            # 检测Win键释放
            elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_r:
                self.win_pressed = False
                #logger.debug("Win键已释放")
                #print("✅ Win键状态: 已释放")
            # 打印当前修饰键状态
            #logger.debug(f"当前修饰键状态 - Ctrl: {self.ctrl_pressed}, Alt: {self.alt_pressed}, Shift: {self.shift_pressed}")
            #print(f"当前修饰键状态 - Ctrl: {self.ctrl_pressed}, Alt: {self.alt_pressed}, Shift: {self.shift_pressed}")
        except Exception as e:
            logger.error(f"处理按键释放事件时出错: {e}")
            #print(f"❌ 错误: {e}")
    
    async def start_vts_model(self):
        """启动VTS模型"""
        print("开始VTubeStudio...")
        await self.vts_controller.connect()
        print("VTubeStudio已启动")
        self.vts_expressions = await self.vts_controller.get_expressions()
        if self.vts_expressions:
            print("VTubeStudio表情列表:", self.vts_expressions)
        else:
            print("未获取到VTubeStudio表情列表")
        
    def get_chat(self):
        """获取聊天消息"""
        while True:
            if chat.msg:
                print(chat.msg)
                self.process_input(chat.msg)
                chat.msg = ""
            time.sleep(1)
    
    async def run(self):
        """运行系统"""
        print("正在初始化系统...")   
        
        if not self.initialize():
            print("系统初始化失败，请检查错误信息")
            return
        
        # 初始问候
        current_datetime = datetime.now().strftime("%H:%M:%S")
        print(f"当前时间: {current_datetime}")
        
        hour = int(current_datetime[:2])
        if hour < 13:
            greeting = "早上好"
        elif hour < 20:
            greeting = "下午好"
        else:
            greeting = "晚上好"
        
        print("=== AI VTuber 交互模式 ===")
        print("输入 'quit' 退出")
        print("输入 'voice' 启动语音模式")
        print("输入 'text' 切换回文本模式")
        print("输入 'history' 查看对话历史")
        print("输入 'clear' 清空对话历史")
        print("输入 '启用鼠标控制' 开启电脑控制功能")
        print("输入 '唱歌' 或 '唱首歌' 让AI演唱歌曲")
        print("输入 '評價' 或 '評價如何' 查看AI對於截图的评价")
        print("输入 'ai控制' 启动AI自动控制模式")
        print("输入 '停止ai控制' 停止AI自动控制模式")
        print("快捷键: Ctrl+Alt+R 切换语音模式")
        print("=======================")
        
        # 导入pygame用于键盘监听
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((1, 1),pygame.NOFRAME)
        SetWindowPos(pygame.display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0001)
        pygame.display.set_caption("AI VTuber 键盘监听")
        
        obs_controller.set_text_source_content("textai123", greeting+"!")
        obs_controller.set_text_source_content("textai", greeting+"!")
        textAI.text("textai",greeting+"!")
        self.tts_speaker.speak(greeting+"!")
        obs_controller.set_text_source_content("textai123", "")
        obs_controller.set_text_source_content("textai", "")
        textAI.text("textai","")
        
        # 设置图标（可选）
        try:
            icon = pygame.Surface((32, 32))
            icon.fill((0, 0, 0))
            pygame.display.set_icon(icon)
        except:
            pass
        
        # 设置键盘监听器，同时监听按下和释放事件
        logger.info("初始化键盘监听器...")
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
            suppress=False  # 不抑制系统按键事件
        )
        self.listener.daemon = True  # 设置为守护线程，主程序退出时自动终止
        self.listener.start()  # 启动键盘监听器
        logger.info("键盘监听器已成功启动")
        print("\n===== 键盘快捷键提示 =====")
        print("按下 Ctrl+Alt+R 组合键可切换语音模式")
        print("当前语音模式状态: {}" .format("开启" if self.is_voice_mode else "关闭"))
        print("======================\n")
        # 确保在使用前导入os模块
        import os
        # 获取Twitch相关环境变量
        twitch_oauth_token = os.getenv('TWITCH_OAUTH_TOKEN', 'ubv7jqnb05tplgblp8zqptiu9j87p1')
        # 使用通用用户名和默认频道
        threading.Thread(target=chat.main).start()
        threading.Thread(target=self.get_chat).start()
        #threading.Thread(target=sub.main).start()
        logger.info("聊天室监听器已启动")
        # 导入time模块
        import time
        import sys
        
        try:
            while True:
                # 检查键盘事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                    elif event.type == pygame.KEYDOWN:
                        # 检查Esc键退出
                        if event.key == pygame.K_ESCAPE:
                            break
                        # 检查Ctrl+Alt+R组合键
                        if pygame.key.get_mods() & pygame.KMOD_CTRL and pygame.key.get_mods() & pygame.KMOD_ALT and event.key == pygame.K_r:
                            self.toggle_voice_mode()
                
                if not self.is_voice_mode:
                    try:
                        # 根据不同操作系统使用不同的非阻塞输入方法
                        if os.name == 'nt':  # Windows系统
                            import msvcrt
                            if msvcrt.kbhit():
                                # 先显示'你：'作为前缀
                                print("你：", end="", flush=True)
                                # 然后读取用户输入
                                user_input = sys.stdin.readline().strip()
                                
                                if user_input.lower() in ['quit', 'exit', 'q']:
                                    break
                                elif user_input.lower() == 'voice':
                                    self.start_voice_mode()
                                elif user_input.lower() == 'history':
                                    history = self.ai_brain.get_history()
                                    for msg in history:
                                        role = "用户" if msg["role"] == "user" else "AndyL"
                                        print(f"{role}: {msg['content']}")
                                elif user_input.lower() == 'clear':
                                    self.ai_brain.clear_history()
                                    print("对话历史已清空")
                                elif user_input.lower() == 'vts on':
                                    print("VTS 模型已启动")
                                    # 使用asyncio.create_task在当前事件循环中运行异步函数
                                    asyncio.create_task(self.start_vts_model())
                                    
                                elif user_input.lower() == 'ai控制':
                                    self.start_ai_control_mode()
                                elif user_input.lower() == '停止ai控制':
                                    self.stop_ai_control_mode()
                                elif user_input.lower() == 'use_system_voice':
                                    print("正在切换到系统默认语音引擎...")
                                    self.speaker.set_tts_type("edge")
                                    print("已成功切换到系统默认语音引擎")
                                    self.speaker.speak("你好，我现在使用的是系统默认的语音引擎。")
                                elif user_input:
                                    print(f"[调试] 处理文本输入: '{user_input}'")
                                    self.process_input(f"用户: {user_input}")
                    except Exception as e:
                        print(f"[调试] 输入处理异常: {e}")
                        # 打印异常信息以便调试
                        logger.warning(f"非阻塞输入处理异常: {e}")
                
                # 检查输入文件（用于自动化测试）
                if os.path.exists('g:\\AndyL AI v1\\input.txt'):
                    try:
                        with open('g:\\AndyL AI v1\\input.txt', 'r', encoding='utf-8') as f:
                            input_text = f.read().strip()
                        if input_text:
                            print(f"读取到输入文件内容: {input_text}")
                            self.process_input(input_text)
                            # 处理完后删除文件
                            os.remove('g:\\AndyL AI v1\\input.txt')
                    except Exception as e:
                        logger.error(f"读取输入文件失败: {e}")
                
                # 短暂休眠，减少CPU占用
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n再见!")
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
        finally:
            # 退出前清理
            pygame.quit()
            if self.listener:
                self.listener.stop()  # 停止键盘监听器
            self.shutdown()

import argparse

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="AI VTuber系统")
    parser.add_argument(
        "--ai-type", 
        choices=["ollama", "localai"], 
        default="ollama",
        help="选择AI服务类型 (默认: ollama)"
    )
    parser.add_argument(
        "--model", 
        help="指定AI模型名称 (默认: 根据AI类型自动选择)"
    )
    parser.add_argument(
        "--tts-type", 
        choices=["windows", "edge"], 
        help="选择TTS服务类型 (默认: 使用.env配置)"
    )
    parser.add_argument(
        "--stt-type", 
        choices=["windows", "localai", "stt"], 
        help="选择STT服务类型 (默认: 使用.env配置)"
    )
    parser.add_argument(
        "--live2d-type", 
        choices=["web", "direct", "vtube_studio"], 
        help="选择Live2D后端类型 (默认: 使用.env配置)"
    )
    parser.add_argument(
        "--live2d-model", 
        help="指定Live2D模型文件路径 (仅direct模式使用)"
    )
    args = parser.parse_args()
    
    # 设置异步事件循环策略
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except:
        pass
    
    # 根据参数创建VTuber系统
    vtuber_system = VTuberSystem(
        ai_type=args.ai_type, 
        ai_model=args.model,
        tts_type=args.tts_type,
        stt_type=args.stt_type,
        live2d_type=args.live2d_type,
        live2d_model_path=args.live2d_model
    )
    asyncio.run(vtuber_system.run())

if __name__ == "__main__":
    main()
