import requests
import json
import logging
import os
import requests
import ollama
from typing import Generator, Dict, List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AIBrain")

class OllamaDeepSeekClient:
    """Ollama DeepSeek客户端 - AI大脑"""
    
    def __init__(self, host=None, port=None, model=None):
        # 从环境变量获取配置，没有则使用默认值
        self.host = host or os.getenv("OLLAMA_HOST", "localhost")
        self.port = port or os.getenv("OLLAMA_PORT", "11434")
        if isinstance(self.port, str):
            self.port = int(self.port)
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = requests.Session()
        self.conversation_history = []
        self.max_history = 10  # 最大对话历史记录数
    
    def check_connection(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"连接Ollama失败: {e}")
            return False
    
    def add_to_history(self, role: str, content: str):
        """添加对话到历史记录"""
        self.conversation_history.append({"role": role, "content": content})
        # 保持历史记录长度
        if len(self.conversation_history) > self.max_history * 2:  # 用户和AI各占一半
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """生成响应（阻塞方式）"""
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            response = self.session.post(url, json=data, timeout=60)
            if response.status_code != 200:
                logger.error(f"生成响应失败: HTTP {response.status_code}")
                return f"错误: HTTP {response.status_code}"
            
            result = response.json()
            response_text = result.get("response", "无响应")
            
            # 添加到历史记录
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"生成响应时出错: {e}")
            return f"错误: {str(e)}"
    
    def generate_response_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """流式生成响应"""
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            response = self.session.post(url, json=data, stream=True, timeout=60)
            if response.status_code != 200:
                error_msg = f"流式生成失败: HTTP {response.status_code}"
                logger.error(error_msg)
                yield error_msg
                return
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        if 'response' in data:
                            chunk = data['response']
                            full_response += chunk
                            yield chunk
                        
                        if data.get('done', False):
                            logger.info(f"响应完成: {len(full_response)}字符")
                            # 添加到历史记录
                            self.add_to_history("user", prompt)
                            self.add_to_history("assistant", full_response)
                            break
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"处理流式响应时出错: {e}")
                        break
        
        except Exception as e:
            error_msg = f"流式请求时出错: {str(e)}"
            logger.error(error_msg)
            yield error_msg
    
    def chat(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """聊天模式"""
        url = f"{self.base_url}/api/chat"
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = self.session.post(url, json=data, timeout=60)
            if response.status_code != 200:
                logger.error(f"聊天请求失败: HTTP {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
            
            result = response.json()
            if "message" in result and "content" in result["message"]:
                # 更新历史记录（只保存用户和助理的对话）
                user_message = next((msg for msg in messages if msg["role"] == "user"), None)
                if user_message:
                    self.add_to_history("user", user_message["content"])
                    self.add_to_history("assistant", result["message"]["content"])
            
            return result
            
        except Exception as e:
            logger.error(f"聊天请求时出错: {e}")
            return {"error": str(e)}
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()

class VTuberPersona:
    """VTuber角色设定"""
    
    def __init__(self, name: str, personality: str, style: str):
        self.name = name
        self.personality = personality
        self.style = style
    
    def create_system_prompt(self) -> str:
        """创建系统提示词"""
        return f"""你是{self.name}，一个虚拟主播(VTuber)。以下是你的设定：

个性: {self.personality}
风格: {self.style}

请以{self.name}的身份和风格回应用户。保持角色一致（有时候可以不用保持角色），回应自然有趣。还有一点，用户叫L。
"""


class AIBrain:
    """AI大脑 - 主处理类"""
    
    def __init__(self, model: str = "deepseek-r1:1.5b", ai_type: str = "ollama"):
        """
        初始化AI大脑
        
        参数:
            model: 模型名称
            ai_type: AI类型，可选值: "ollama", "localai"
        """
        self.ai_type = ai_type
        self.persona = None
        self.system_prompt = None
        
        if ai_type == "localai":
            self.client = LocalAIClient(model=model)
        else:
            # 默认使用Ollama
            self.client = OllamaDeepSeekClient(model=model)
    
    def initialize(self, name: str, personality: str, style: str) -> bool:
        """初始化AI大脑"""
        # 检查连接
        if not self.client.check_connection():
            logger.error(f"无法连接到{self.ai_type}服务")
            return False
        
        # 创建角色
        self.persona = VTuberPersona(name, personality, style)
        self.system_prompt = self.persona.create_system_prompt()
        
        logger.info(f"AI大脑初始化完成: {name} (使用{self.ai_type})")
        return True
    
    def process_text(self, text: str, use_stream: bool = False) -> str:
        """处理文本输入"""
        if not self.persona:
            return "AI大脑未初始化"
        
        if use_stream:
            # 流式处理 - 返回生成器
            return self.client.generate_response_stream(text, self.system_prompt)
        else:
            # 阻塞处理 - 返回完整文本
            return self.client.generate_response(text, self.system_prompt)
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.client.get_conversation_history()
    
    def clear_history(self):
        """清空对话历史"""
        self.client.clear_history()


# 单例实例
ai_brain_instance = None

def get_ai_brain(model: str = "llama3.2:latest", ai_type: str = "ollama") -> AIBrain:
    """获取AI大脑实例（单例模式）"""
    global ai_brain_instance
    if ai_brain_instance is None:
        ai_brain_instance = AIBrain(model=model, ai_type=ai_type)
    return ai_brain_instance