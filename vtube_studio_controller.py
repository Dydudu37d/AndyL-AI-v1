import os
import json
import asyncio
import logging
import time
from datetime import datetime
import inspect
import re
import uuid
from traceback import print_exception
from typing import List, Dict, Optional, Any, Union, Tuple, Literal, Callable

from check_vts_class import request_sig

# 尝试导入PyTubeStudio库
HAS_PYTUBESTUDIO = False
try:
    from PyTubeStudio.client import *
    import VtsModels.models as models
    HAS_PYTUBESTUDIO = True
except ImportError:
    logging.warning("PyTubeStudio library not available")

# 尝试导入pyvts库
HAS_PYVTS = False
try:
    import pyvts
    HAS_PYVTS = True
except ImportError:
    logging.warning("pyvts library not available")

# 如果两个库都不可用，抛出错误
if not HAS_PYTUBESTUDIO and not HAS_PYVTS:
    raise ImportError("Neither PyTubeStudio nor pyvts library is available. Please install at least one.")

# 设置日志，修复中文编码问题
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vtube_studio_controller.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VTubeStudioController")

# 确保中文能正常显示
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

class VTubeStudioController:
    def __init__(self, port:int = 8001, ws_ip:str = "localhost", token_path:str = "./pyvts_token.txt", 
                 models_dir: str = None, library: str = "auto"):
        """
        初始化VTube Studio控制器
        
        参数:
            port: VTube Studio API端口
            ws_ip: VTube Studio API IP地址
            token_path: 认证token保存路径
            models_dir: 模型目录路径
            library: 使用的库，可选值: "auto", "pytubestudio", "pyvts"
        """
        self.port = port
        self.ws_ip = ws_ip
        self.token_path = token_path
        self.connected = False
        self.is_authenticated = False
        self.vts_client = None
        self.library_type = ""
        
        # 设置默认的模型目录
        if models_dir is None:
            # Windows 上的默认模型目录
            documents_path = os.path.expanduser("~")
            self.models_dir = os.path.join(
                documents_path, "Documents", "VTube Studio", "Models"
            )
        else:
            self.models_dir = models_dir
        
        # 初始化客户端库
        if library == "auto":
            # 自动选择可用的库，优先使用pyvts
            if HAS_PYVTS:
                self._init_pyvts_client()
            elif HAS_PYTUBESTUDIO:
                self._init_pytubestudio_client()
        elif library == "pyvts" and HAS_PYVTS:
            self._init_pyvts_client()
        elif library == "pytubestudio" and HAS_PYTUBESTUDIO:
            self._init_pytubestudio_client()
        else:
            logger.error(f"Requested library '{library}' is not available. Using auto-selection.")
            if HAS_PYVTS:
                self._init_pyvts_client()
            elif HAS_PYTUBESTUDIO:
                self._init_pytubestudio_client()
        
        logger.info(f"VTube Studio Controller initialized, port: {port}, IP: {ws_ip}, library: {self.library_type}")
        logger.info(f"Using models directory: {self.models_dir}")
        
    def _init_pyvts_client(self):
        """初始化pyvts客户端"""
        try:
            # 增加更详细的plugin_info信息，这可能有助于认证
            self.plugin_info = {
                "plugin_name": "AI_VTuber_Plugin",
                "developer": "AndyL",
                "authentication_token_path": self.token_path,
                "plugin_version": "1.0",
                "requested_api_version": "1.0"
            }
            
            logger.info(f"Initializing pyvts client with plugin_info: {self.plugin_info}")
            self.vts_client = pyvts.vts(plugin_info=self.plugin_info, port=self.port)
            self.library_type = "pyvts"
            logger.info(f"pyvts client initialized, version: {pyvts.__version__}")
            
            # 检查token文件是否存在
            if os.path.exists(self.token_path):
                logger.info(f"Found existing authentication token at {self.token_path}")
            else:
                logger.info(f"No existing authentication token found at {self.token_path}, will generate new one")
        except Exception as e:
            logger.error(f"Failed to initialize pyvts client: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_pytubestudio_client(self):
        """初始化PyTubeStudio客户端"""
        try:
            self.vts_client = PyTubeStudio(port=self.port, ws_ip=self.ws_ip, token_path=self.token_path)
            self.library_type = "pytubestudio"
            logger.info("PyTubeStudio client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PyTubeStudio client: {e}")

    async def get_models(self) -> List[Dict[str, Any]]:
        """获取所有模型信息"""
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "AvailableModelsRequest",
                "data": None
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="AvailableModelsRequest",
                data=None
            )

        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio get models response: {response}")
        logger.info(f"Response type: {type(response)}")
        
        try:
            # 检查响应类型和结构
            if hasattr(response, 'data') and hasattr(response.data, 'availableModels'):
                # 如果是对象类型，使用对象属性访问
                models_data = response.data.availableModels
                if models_data:
                    logger.info(f"Successfully retrieved {len(models_data)} models")
                    # 转换为字典列表
                    return [model.__dict__ if hasattr(model, '__dict__') else dict(model) for model in models_data]
            elif isinstance(response, dict) and 'data' in response and 'availableModels' in response['data']:
                # 如果是字典类型，使用字典键访问
                models_data = response['data']['availableModels']
                if models_data:
                    logger.info(f"Successfully retrieved {len(models_data)} models")
                    return list(models_data)
            
            # 尝试JSON解析
            import json
            if isinstance(response, str):
                response_dict = json.loads(response)
                if 'data' in response_dict and 'availableModels' in response_dict['data']:
                    models_data = response_dict['data']['availableModels']
                    if models_data:
                        logger.info(f"Successfully retrieved {len(models_data)} models")
                        return list(models_data)
            
            logger.warning(f"VTube Studio get models response format not recognized")
            return []
        except Exception as e:
            logger.error(f"Error processing VTube Studio models response: {e}")
            return []

    async def load_model(self,model_id:str):
        """加载模型"""
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "ModelLoadRequest",
                "data": {
                    "modelID": model_id
                }
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="ModelLoadRequest",
                data={
                    "modelID": model_id
                }
            )
            
        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio load model response: {response}")
        
        # 处理不同类型的响应
        if isinstance(response, dict):
            if response.get("messageType") == "ModelLoadResponse":
                logger.info(f"VTube Studio load model success, modelID: {model_id}")
            else:
                logger.error(f"VTube Studio load model failed: {response}")
        elif hasattr(response, 'message_type') and response.message_type == "ModelLoadResponse":
            logger.info(f"VTube Studio load model success, modelID: {model_id}")
        else:
            logger.error(f"VTube Studio load model failed: {response}")

    async def get_hotkeys(self,model_id:str = None) -> List[Dict[str, Any]]:
        """获取当前模型的热键"""
        
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "HotkeysInCurrentModelRequest",
                "data": {
                    "modelID": model_id
                }
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="HotkeysInCurrentModelRequest",
                data={
                    "modelID": model_id
                }
            )
        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio get hotkeys response: {response}")
        logger.info(f"Response type: {type(response)}")
        
        # 处理字符串类型的响应
        if isinstance(response, str):
            import json
            try:
                response_dict = json.loads(response)
                if response_dict.get("messageType") == "HotkeysInCurrentModelResponse":
                    return response_dict.get("data", {}).get("availableHotkeys", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse response as JSON: {response}")
        
        # 处理字典类型的响应
        elif isinstance(response, dict):
            if response.get("message_type") == "HotkeysInCurrentModelResponse" or response.get("messageType") == "HotkeysInCurrentModelResponse":
                return response.get("data", {}).get("availableHotkeys", [])
        
        # 处理对象类型的响应
        elif hasattr(response, 'message_type') and response.message_type == "HotkeysInCurrentModelResponse":
            if hasattr(response, 'data') and hasattr(response.data, 'availableHotkeys'):
                return response.data.availableHotkeys
        
        logger.error(f"VTube Studio get hotkeys failed: {response}")
        return []
    
    async def get_parameters(self):
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "InputParameterListRequest",
                "data": None
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="InputParameterListRequest",
                data=None
            )
        
        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio get parameters response: {response}")
        
        # 处理响应
        if isinstance(response, dict) and response.get("messageType") == "InputParameterListResponse":
            return response.get("data", {})
        elif hasattr(response, 'message_type') and response.message_type == "InputParameterListResponse":
            return response.data
        
        return {}
        
    
    async def get_available_parameters(self) -> List[Dict[str, Any]]:
        """获取当前模型的可用参数"""
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "AvailableParametersRequest",
                "data": None
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="AvailableParametersRequest",
                data=None
            )
        
        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio get available parameters response: {response}")
        
        # 处理响应
        if isinstance(response, dict) and response.get("messageType") == "AvailableParametersResponse":
            return response.get("data", {}).get("availableParameters", [])
        elif hasattr(response, 'message_type') and response.message_type == "AvailableParametersResponse" and hasattr(response, 'data') and hasattr(response.data, 'availableParameters'):
            return response.data.availableParameters
        
        logger.error(f"VTube Studio get available parameters failed: {response}")
        return []
        
    
    async def get_vts_folders(self):
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "VTSFolderInfoRequest",
                "data": None
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="VTSFolderInfoRequest"
            )
        
        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio get vts folders response: {response}")
        
        # 处理不同类型的响应
        if isinstance(response, dict) and response.get("messageType") == "VTSFolderInfoResponse":
            return response.get("data", {})
        elif hasattr(response, 'message_type') and response.message_type == "VTSFolderInfoResponse":
            return response.data
        
        return {}
    
    async def get_current_model(self) -> Dict[str, Any]:
        """获取当前加载的模型"""
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "CurrentModelRequest",
                "data": None
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="CurrentModelRequest",
                data=None
            )
        
        try:
            response = await self.vts_client.request(request_msg)
            logger.info(f"VTube Studio get current model response: {response}")
            logger.info(f"Response type: {type(response)}")
            
            # 处理字符串类型的响应
            if isinstance(response, str):
                import json
                try:
                    response_dict = json.loads(response)
                    if response_dict.get("messageType") == "CurrentModelResponse":
                        # 直接返回data部分，不需要再次JSON解析
                        return response_dict.get("data", {})
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response as JSON: {response}")
            
            # 处理字典类型的响应
            elif isinstance(response, dict):
                if response.get("message_type") == "CurrentModelResponse" or response.get("messageType") == "CurrentModelResponse":
                    return response.get("data", {})
            
            # 处理对象类型的响应
            elif hasattr(response, 'message_type') and response.message_type == "CurrentModelResponse":
                if hasattr(response, 'data'):
                    return response.data
            
            logger.error(f"VTube Studio get current model failed with unexpected response format: {response}")
            return {}
        except Exception as e:
            logger.error(f"VTube Studio get current model exception: {e}")
            return {}
    
    async def close_all_expression(self):
        expressions = await self.get_available_parameters()
        for expression in expressions:
            if expression.get('type') == 'expression':
                await self.set_expression(expression.get('name', ''), False)
    
    async def activation_hotkey(self, hotkey_name: str|int):
        """激活热键"""
        # 检查连接状态
        if not self.connected:
            logger.error("Not connected to VTube Studio")
            return False
        
        try:
            # 根据不同的库类型调用不同的热键激活方法
            if self.library_type == "pyvts":
                return await self._activation_hotkey_pyvts(hotkey_name)
            elif self.library_type == "pytubestudio" and HAS_PYTUBESTUDIO:
                return await self._activation_hotkey_pytubestudio(hotkey_name)
            else:
                logger.error(f"Unsupported library type: {self.library_type}")
                return False
        except Exception as e:
            logger.error(f"Error activating hotkey '{hotkey_name}': {e}")
            return False
            
    async def _activation_hotkey_pyvts(self, hotkey_name: str|int):
        """使用pyvts库激活热键"""
        try:
            # 先获取当前模型信息
            current_model = await self.get_current_model()
            model_id = current_model.get("modelID", "") if current_model else ""
            
            logger.info(f"Activating hotkey '{hotkey_name}' with pyvts, model ID: '{model_id}'")
            
            # 构建热键触发请求 - pyvts库需要使用驼峰命名格式
            request_data = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "HotkeyTriggerRequest",
                "data": {
                    "hotkeyID": str(hotkey_name),
                    "active": True,
                    "itemInstanceID": model_id
                }
            }
            
            # 发送请求
            response = await self.vts_client.request(request_data)
            
            # 处理响应
            if not response or "data" not in response:
                logger.error(f"Invalid response for hotkey '{hotkey_name}'")
                return False
            
            data = response["data"]
            if "apiName" in data and data["apiName"] == "APIError":
                # 处理API错误
                error_id = data["data"].get("errorID", "unknown")
                error_msg = data["data"].get("message", "unknown error")
                logger.error(f"API Error {error_id}: {error_msg}")
                
                if error_id == "8" or ("code" in data["data"] and data["data"]["code"] == 8):
                    # 错误码8表示未认证，尝试重新认证
                    logger.warning("Session not authenticated. Trying to reauthenticate...")
                    
                    # 尝试重新认证
                    try:
                        await self.vts_client.request_authenticate()
                        self.is_authenticated = True
                        logger.info("Reauthentication successful")
                        
                        # 重新发送请求
                        response = await self.vts_client.request(request_data)
                        
                        if not response or "data" not in response:
                            logger.error(f"Invalid response after reauthentication for hotkey '{hotkey_name}'")
                            return False
                        
                        data = response["data"]
                        if "apiName" in data and data["apiName"] == "APIError":
                            logger.error(f"Retry failed with API Error: {data["data"]}")
                            return False
                    except Exception as e:
                        logger.error(f"Reauthentication failed: {e}")
                        return False
                else:
                    return False
            
            # 检查热键是否激活成功
            if "apiName" in data and data["apiName"] == "HotkeyTriggerResponse":
                logger.info(f"Hotkey '{hotkey_name}' activated successfully with pyvts")
                return True
            else:
                logger.warning(f"Unexpected response format for hotkey '{hotkey_name}': {data}")
                
                # 尝试调试热键列表
                await self._debug_hotkeys()
                return False
        except Exception as e:
            logger.error(f"Error activating hotkey '{hotkey_name}' with pyvts: {e}")
            
            # 尝试调试热键列表
            try:
                await self._debug_hotkeys()
            except Exception as debug_e:
                logger.error(f"Error debugging hotkeys: {debug_e}")
            
            return False
            
    async def _activation_hotkey_pytubestudio(self, hotkey_name: str|int):
        """使用PyTubeStudio库激活热键"""
        # 先获取当前模型信息
        current_model = await self.get_current_model()
        model_id = current_model.get("modelID", "") if current_model else ""
        
        logger.info(f"Activating hotkey '{hotkey_name}' with PyTubeStudio, model ID: '{model_id}'")
        
        # 首先尝试使用旧的HotkeyTriggerRequest格式（这是更通用的格式）
        legacy_request = models.BaseRequest(
            api_name="VTubeStudioPublicAPI",
            api_version="1.0",
            request_id=str(uuid.uuid4()),
            message_type="HotkeyTriggerRequest",
            data={
                "hotkeyID": hotkey_name,
                "itemInstanceID": model_id,
            }
        )
        
        logger.debug(f"Hotkey trigger request data (Legacy HotkeyTriggerRequest format): {legacy_request}")
        
        try:
            response = await self.vts_client.request(legacy_request)
            logger.debug(f"VTube Studio activation hotkey response: {response}")
            logger.debug(f"Response type: {type(response)}")
            
            # 处理不同类型的响应
            success = False
            
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response)
                    # 检查是否为成功响应
                    if response_dict.get("messageType") == "APIError":
                        error_id = response_dict.get("data", {}).get("errorID", "unknown")
                        error_msg = response_dict.get("data", {}).get("message", "unknown error")
                        logger.error(f"API Error {error_id}: {error_msg}")
                        
                        # 如果是未认证错误，尝试重新认证
                        if error_id == "8":  # APIError 8表示未认证
                            logger.warning("Session not authenticated. Trying to reauthenticate...")
                            try:
                                # PyTubeStudio的authenticate方法不返回值，所以这里只能尝试执行
                                await self.vts_client.authenticate()
                                self.is_authenticated = True
                                logger.info("Reauthentication attempted. Retrying hotkey activation...")
                                # 重新发送请求
                                response = await self.vts_client.request(legacy_request)
                                logger.debug(f"Retry response: {response}")
                                # 重新解析响应
                                if isinstance(response, str):
                                    try:
                                        response_dict = json.loads(response)
                                        if response_dict.get("messageType") == "APIError":
                                            logger.error(f"Retry failed with API Error {response_dict.get('data', {}).get('errorID', 'unknown')}: {response_dict.get('data', {}).get('message', 'unknown error')}")
                                        elif response_dict.get("messageType") == "HotkeyTriggerResponse":
                                            success = True
                                            logger.info(f"Hotkey '{hotkey_name}' activated successfully after reauthentication")
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse retry response as JSON")
                            except Exception as e:
                                logger.error(f"Reauthentication failed: {e}")
                    elif response_dict.get("messageType") == "HotkeyTriggerResponse":
                        success = True
                        logger.info(f"VTube Studio activation hotkey success, hotkeyID: {hotkey_name}")
                    else:
                        logger.warning(f"Unexpected response message type: {response_dict.get('messageType')}")
                        logger.debug(f"Full response: {json.dumps(response_dict, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response as JSON: {response}")
            elif isinstance(response, dict):
                if response.get("messageType") == "HotkeyTriggerResponse":
                    success = True
                    logger.info(f"VTube Studio activation hotkey success, hotkeyID: {hotkey_name}")
                else:
                    logger.warning(f"Unexpected dict response structure: {response}")
            elif hasattr(response, 'message_type') and response.message_type == "HotkeyTriggerResponse":
                success = True
                logger.info(f"VTube Studio activation hotkey success, hotkeyID: {hotkey_name}")
            else:
                # 对于不匹配上述条件的响应，提供更详细的调试信息
                logger.warning(f"Unexpected response format or type. Response: {response}")
                logger.warning(f"Response type: {type(response)}")
                # 尝试以字符串形式查看响应内容
                if not isinstance(response, str):
                    try:
                        logger.warning(f"String representation of response: {str(response)}")
                    except:
                        pass
            
            if success:
                return True
            else:
                logger.error(f"VTube Studio activation hotkey failed")
        except Exception as e:
            logger.error(f"Exception during hotkey activation: {e}")
            import traceback
            traceback.print_exc()
        
        # 如果都失败了，尝试列出所有可用热键以帮助调试
        try:
            hotkeys = await self.get_hotkeys()
            if hotkeys:
                logger.info(f"Available hotkeys for debugging: {[{'name': hk.get('name'), 'hotkeyID': hk.get('hotkeyID')} for hk in hotkeys]}")
                # 检查我们尝试激活的热键是否在列表中
                found = False
                for hk in hotkeys:
                    if hk.get('hotkeyID') == str(hotkey_name) or hk.get('name') == str(hotkey_name):
                        found = True
                        logger.info(f"Found hotkey in available list: {hk}")
                        break
                if not found:
                    logger.warning(f"Hotkey '{hotkey_name}' not found in available hotkeys list")
            else:
                logger.warning("No hotkeys found in the model")
        except Exception as e:
            logger.error(f"Failed to get hotkeys list for debugging: {e}")
        
        # 最后尝试一个简单的热键请求，不涉及认证和模型ID，仅用于测试连接
        try:
            simple_request = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="APIStateRequest",
                data=None
            )
            simple_response = await self.vts_client.request(simple_request)
            logger.debug(f"API State Test Response: {simple_response}")
        except Exception as e:
            logger.error(f"API State Test failed: {e}")
        
        return False
    
    async def add_parameters(self,parameter_name:str,explanation:str,min:int,max:int,default_value:float):
        """添加参数"""
        # 根据不同的库类型构建不同格式的请求
        if self.library_type == "pyvts" and HAS_PYVTS:
            # pyvts库需要字典格式的请求
            request_msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "ParameterCreationRequest",
                "data": {
                    "parameterName": parameter_name,
                    "explanation": explanation,
                    "min": min,
                    "max": max,
                    "defaultValue": default_value,
                }
            }
        else:
            # PyTubeStudio库使用BaseRequest对象
            request_msg = models.BaseRequest(
                api_name="VTubeStudioPublicAPI",
                api_version="1.0",
                request_id=str(uuid.uuid4()),
                message_type="ParameterCreationRequest",
                data={
                    "parameterName": parameter_name,
                    "explanation": explanation,
                    "min": min,
                    "max": max,
                    "defaultValue": default_value,
                }
            )
        
        response = await self.vts_client.request(request_msg)
        logger.info(f"VTube Studio add parameters response: {response}")
        
        # 处理响应
        if isinstance(response, dict) and response.get("messageType") == "AddParametersResponse":
            logger.info(f"VTube Studio add parameters success, parameterName: {parameter_name}")
        elif hasattr(response, 'message_type') and response.message_type == "AddParametersResponse":
            logger.info(f"VTube Studio add parameters success, parameterName: {parameter_name}")
        else:
            logger.error(f"VTube Studio add parameters failed: {response}")
        
    
    async def get_model_path(self,model_id:str) -> str:
        """获取模型路径"""
        model = await self.get_current_model()
        vts_path = await self.get_vts_folders()
        if model and model.get("modelID") == model_id:
            return str(vts_path.get("data", "").get("models", "")+"/"+model.get("data", "").get("modelName", "")+"_vts")
        return ""
    
    async def get_expressions(self) -> List[Dict[str, Any]]:
        """获取当前模型的所有可用表情列表"""
        try:
            # 首先检查vts_client是否存在且已连接
            if self.vts_client is None:
                logger.error("vts_client is None! Please initialize and connect first.")
                return []
            
            if not self.connected:
                logger.warning("Not connected to VTube Studio! Attempting to connect...")
                await self.connect()
                if not self.connected:
                    logger.error("Failed to connect to VTube Studio")
                    return []
            
            # 根据不同的库类型构建不同格式的请求
            if self.library_type == "pyvts" and HAS_PYVTS:
                # pyvts库需要字典格式的请求
                request_msg = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": str(uuid.uuid4()),
                    "messageType": "ExpressionStateRequest",
                    "data": None
                }
            else:
                # PyTubeStudio库使用BaseRequest对象
                request_msg = models.BaseRequest(
                    api_name="VTubeStudioPublicAPI",
                    api_version="1.0",
                    request_id=str(uuid.uuid4()),
                    message_type="ExpressionStateRequest",
                    data=None
                )
            
            response = await self.vts_client.request(request_msg)
            logger.info(f"VTube Studio get expressions response: {response}")
            logger.info(f"Response type: {type(response)}")
            
            # 处理字符串类型的响应
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response)
                    if response_dict.get("messageType") == "ExpressionStateResponse":
                        expressions = response_dict.get("data", {}).get("expressions", [])
                        logger.info(f"Successfully got {len(expressions)} expressions from API")
                        return expressions
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response as JSON: {response}")
            
            # 处理字典类型的响应
            elif isinstance(response, dict):
                if response.get("message_type") == "ExpressionStateResponse" or response.get("messageType") == "ExpressionStateResponse":
                    expressions = response.get("data", {}).get("expressions", [])
                    logger.info(f"Successfully got {len(expressions)} expressions from API")
                    return expressions
            
            # 处理对象类型的响应
            elif hasattr(response, 'message_type') and response.message_type == "ExpressionStateResponse":
                if hasattr(response, 'data') and hasattr(response.data, 'expressions'):
                    expressions = response.data.expressions
                    logger.info(f"Successfully got {len(expressions)} expressions from API")
                    return expressions
            
            logger.error(f"Failed to get expressions with unexpected response format: {response}")
            return []
        except Exception as e:
            logger.error(f"VTube Studio get expressions exception: {e}")
            return []
    
    async def set_expression(self, expression_name: str, active: bool, fade_time: float = 0.3) -> bool:
        """设置表情状态
        
        Args:
            expression_name: 表情名称或文件名
            active: 是否激活表情
            fade_time: 淡入淡出时间（秒）
        
        Returns:
            bool: 是否设置成功
        """
        try:
            # 首先检查vts_client是否存在且已连接
            if self.vts_client is None:
                logger.error("vts_client is None! Please initialize and connect first.")
                return False
            
            if not self.connected:
                logger.warning("Not connected to VTube Studio! Attempting to connect...")
                await self.connect()
                if not self.connected:
                    logger.error("Failed to connect to VTube Studio")
                    return False
            
            # 检查是否以.exp3.json结尾，如果不是则添加
            if not expression_name.endswith('.exp3.json'):
                expression_name += '.exp3.json'
                logger.info(f"Added .exp3.json extension to expression name: {expression_name}")
            
            # 根据不同的库类型构建不同格式的请求
            if self.library_type == "pyvts" and HAS_PYVTS:
                # pyvts库需要字典格式的请求
                request_msg = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": str(uuid.uuid4()),
                    "messageType": "ExpressionStateRequest",
                    "data": {
                        "expressionName": expression_name,
                        "active": active,
                        "fadeTime": fade_time  # 淡入淡出时间
                    }
                }
            else:
                # PyTubeStudio库使用BaseRequest对象
                request_msg = models.BaseRequest(
                    api_name="VTubeStudioPublicAPI",
                    api_version="1.0",
                    request_id=str(uuid.uuid4()),
                    message_type="ExpressionActivationRequest",
                    data={
                        "expressionFile": expression_name,
                        "active": active,
                        "fadeTime": fade_time  # 淡入淡出时间
                    }
                )
            
            logger.info(f"Sending expression state request: {expression_name}, active: {active}, fade_time: {fade_time}")
            logger.debug(f"Full request message: {vars(request_msg) if hasattr(request_msg, '__dict__') else request_msg}")
            response = await self.vts_client.request(request_msg)
            logger.info(f"VTube Studio set expression response: {response}")
            logger.info(f"Response type: {type(response)}")
            
            # 处理字符串类型的响应
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response)
                    if response_dict.get("messageType") == "ExpressionStateResponse":
                        logger.info(f"Successfully set expression {expression_name} to {'active' if active else 'inactive'}")
                        return True
                    elif response_dict.get("messageType") == "APIError":
                        error_id = response_dict.get("data", {}).get("errorID", "unknown")
                        error_message = response_dict.get("data", {}).get("message", "unknown error")
                        logger.error(f"VTube Studio API error ({error_id}): {error_message}")
                        return False
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response as JSON: {response}")
            
            # 处理字典类型的响应
            elif isinstance(response, dict):
                if response.get("message_type") == "ExpressionStateResponse" or response.get("messageType") == "ExpressionStateResponse":
                    logger.info(f"Successfully set expression {expression_name} to {'active' if active else 'inactive'}")
                    return True
                elif response.get("message_type") == "APIError" or response.get("messageType") == "APIError":
                    error_id = response.get("data", {}).get("errorID", "unknown")
                    error_message = response.get("data", {}).get("message", "unknown error")
                    logger.error(f"VTube Studio API error ({error_id}): {error_message}")
                    return False
            
            # 处理对象类型的响应
            elif hasattr(response, 'message_type'):
                if response.message_type == "ExpressionStateResponse":
                    logger.info(f"Successfully set expression {expression_name} to {'active' if active else 'inactive'}")
                    return True
                elif response.message_type == "APIError":
                    error_info = "unknown"
                    if hasattr(response, 'data'):
                        error_info = response.data
                    logger.error(f"VTube Studio API error: {error_info}")
                    return False
            
            logger.error(f"VTube Studio set expression failed with unexpected response format: {response}")
            return False
        except Exception as e:
            logger.error(f"VTube Studio set expression exception: {e}")
            return False

    async def connect(self):
        """连接到 VTube Studio"""
        try:
            if not self.vts_client:
                logger.error("No VTS client initialized")
                return False
            
            # 添加连接前检查提示
            logger.info("请确保VTube Studio应用程序已经启动，并且在设置中启用了API服务器！")
            logger.info(f"当前尝试连接的端口: {self.port}")
            
            if self.library_type == "pyvts" and HAS_PYVTS:
                # 使用pyvts库连接
                logger.info(f"Connecting to VTube Studio using pyvts on port {self.port}...")
                
                # 检查并处理token文件
                if os.path.exists(self.token_path):
                    # 检查文件大小，如果为0字节，删除它
                    if os.path.getsize(self.token_path) == 0:
                        logger.warning(f"Found empty token file at {self.token_path}, deleting it...")
                        try:
                            os.remove(self.token_path)
                            logger.info("Empty token file deleted")
                        except Exception as e:
                            logger.error(f"Failed to delete empty token file: {e}")
                    else:
                        logger.info(f"Found existing token file at {self.token_path}")
                else:
                    logger.info(f"No existing token file found at {self.token_path}")
                
                try:
                    # pyvts库的connect方法已经包含了连接和状态检查
                    await self.vts_client.connect()
                    self.connected = True
                    logger.info(f"VTube Studio connected, port: {self.port}")
                    
                    # 尝试获取认证token
                    try:
                        logger.info("Requesting authentication token...")
                        auth_token_response = await self.vts_client.request_authenticate_token()
                        logger.info(f"Authentication token response: {auth_token_response}")
                        # 等待一会儿让token保存完成
                        await asyncio.sleep(1.0)  # 增加等待时间，确保token正确保存
                        
                        # 检查token文件是否已创建且不为空
                        if os.path.exists(self.token_path) and os.path.getsize(self.token_path) > 0:
                            logger.info(f"Authentication token saved successfully to {self.token_path}")
                        else:
                            logger.warning(f"Token file {self.token_path} not created or is empty after token request")
                    except Exception as e:
                        logger.warning(f"Failed to get authentication token: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # 尝试进行认证（最多尝试3次）
                    max_auth_attempts = 3
                    for attempt in range(max_auth_attempts):
                        try:
                            logger.info(f"Attempting to authenticate (attempt {attempt+1}/{max_auth_attempts})...")
                            auth_response = await self.vts_client.request_authenticate()
                            logger.info(f"Authentication response: {auth_response}")
                            logger.info(f"Authentication response type: {type(auth_response)}")
                            
                            # 处理pyvts库返回布尔值的情况
                            if isinstance(auth_response, bool):
                                if auth_response:
                                    self.is_authenticated = True
                                    logger.info("VTube Studio authentication successful (boolean response)")
                                    # 认证成功后，验证一下是否真的可以访问API
                                    try:
                                        # 发送一个简单的API测试请求
                                        test_request = {
                                            "apiName": "VTubeStudioPublicAPI",
                                            "apiVersion": "1.0",
                                            "requestID": str(uuid.uuid4()),
                                            "messageType": "APIStateRequest",
                                            "data": None
                                        }
                                        test_response = await self.vts_client.request(test_request)
                                        logger.info(f"API State Test Response: {test_response}")
                                    except Exception as test_e:
                                        logger.warning(f"API Test failed after authentication: {test_e}")
                                    
                                    return True
                                else:
                                    logger.error("Authentication failed (boolean response: False)")
                            # 处理pyvts库返回字典的情况
                            elif isinstance(auth_response, dict):
                                # 检查响应是否包含成功的AuthenticationResponse
                                if auth_response.get("data", {}).get("apiName") == "AuthenticationResponse":
                                    # 检查authenticated字段
                                    if auth_response.get("data", {}).get("data", {}).get("authenticated", False):
                                        self.is_authenticated = True
                                        logger.info("VTube Studio认证成功")
                                        auth_success = True
                                        break
                                    else:
                                        reason = auth_response.get("data", {}).get("data", {}).get("reason", "unknown")
                                        logger.error(f"认证失败: {reason}")
                                elif auth_response.get("messageType") == "APIError":
                                    error_id = auth_response.get("data", {}).get("errorID", "unknown")
                                    error_msg = auth_response.get("data", {}).get("message", "unknown error")
                                    logger.error(f"认证API错误 {error_id}: {error_msg}")
                            else:
                                logger.error(f"认证响应不是字典或布尔值: {type(auth_response)}")
                                logger.error(f"完整认证响应: {auth_response}")
                            
                        except Exception as e:
                            logger.error(f"认证尝试 {attempt+1} 失败: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # 如果不是最后一次尝试，等待一会儿再重试
                        if attempt < max_auth_attempts - 1:
                            logger.info("等待1秒后重试...")
                            await asyncio.sleep(1)
                        
                    if not auth_success:
                        self.is_authenticated = False
                        logger.error("多次尝试后VTube Studio认证失败")
                        
                        # 如果token文件存在，尝试删除它并提示用户重新授权
                        if os.path.exists(self.token_path):
                            logger.warning(f"认证token {self.token_path} 无效或已被撤销。")
                            logger.warning("删除无效的token文件...")
                            try:
                                os.remove(self.token_path)
                                logger.info("无效token文件已删除。请重新运行程序生成新的token。")
                                logger.info("注意: 重新运行时，VTube Studio会显示授权对话框 - 请点击'允许'。")
                            except Exception as e:
                                logger.error(f"删除无效token文件失败: {e}")
                        
                        # 认证失败时返回False，因为没有认证的连接是无法使用的
                        return False
                    
                    return True
                except Exception as e:
                    logger.error(f"VTube Studio连接失败: {e}")
                    if '拒绝网络连接' in str(e) or 'ConnectionRefused' in str(e):
                        logger.error("连接被拒绝！请确保VTube Studio已启动并启用了API服务器。")
                        logger.error("步骤1: 启动VTube Studio应用程序")
                        logger.error("步骤2: 在VTube Studio设置中找到'API设置'，启用'启用API'")
                        logger.error(f"步骤3: 检查端口号是否与代码中的端口({self.port})匹配")
                    self.connected = False
                    return False
            elif self.library_type == "pytubestudio" and HAS_PYTUBESTUDIO:
                # 使用PyTubeStudio库连接
                logger.info(f"正在使用PyTubeStudio连接VTube Studio (端口: {self.port})...")
                await self.vts_client.connect()
                
                # 检查连接是否真的成功（websocket不为None且connected为True）
                if hasattr(self.vts_client, 'websocket') and self.vts_client.websocket is None or not hasattr(self.vts_client, 'connected') or not self.vts_client.connected:
                    logger.error(f"VTube Studio连接已建立但websocket为None或未连接")
                    self.connected = False
                    return False
                
                self.connected = True
                logger.info(f"VTube Studio已连接，端口: {self.port}, IP: {self.ws_ip}")
                
                # 认证客户端
                try:
                    await self.vts_client.authenticate()
                    self.is_authenticated = True
                    logger.info("VTube Studio认证成功")
                except Exception as e:
                    logger.error(f"VTube Studio认证失败: {e}")
                    # 即使认证失败，连接仍然可能是可用的，所以保持connected为True
                
                return True
            else:
                logger.error(f"未知的库类型: {self.library_type}")
                return False
                
        except Exception as e:
            logger.error(f"VTube Studio连接失败: {e}")
            self.connected = False
            self.is_authenticated = False
            return False

    async def disconnect(self) -> None:
        """断开与 VTube Studio 的连接"""
        if self.connected:
            try:
                # 根据不同的库类型执行不同的断开连接操作
                if self.library_type == "pyvts":
                    await self.vts_client.close()
                elif self.library_type == "pytubestudio" and HAS_PYTUBESTUDIO:
                    # 安全地关闭连接，检查websocket是否存在
                    if hasattr(self.vts_client, 'websocket') and self.vts_client.websocket is not None:
                        await self.vts_client.close()
                
                self.connected = False
                self.is_authenticated = False
                logger.info(f"VTube Studio已断开连接，端口: {self.port}, 库: {self.library_type}")
            except Exception as e:
                logger.error(f"断开连接异常: {e}")
                # 即使关闭连接出错，也将连接状态设置为False
                self.connected = False
                self.is_authenticated = False

    async def _debug_hotkeys(self) -> None:
        """调试热键列表"""
        try:
            current_model = await self.get_current_model()
            model_id = current_model.get("modelID", "") if current_model else ""
            logger.info(f"当前模型ID: {model_id}")
            
            hotkeys = await self.get_hotkeys(model_id)
            if hotkeys:
                logger.info(f"调试 - 可用热键数量: {len(hotkeys)}")
                for idx, hotkey in enumerate(hotkeys):
                    hotkey_id = hotkey.get('hotkeyID', 'unknown')
                    hotkey_name = hotkey.get('name', 'unknown')
                    logger.info(f"调试 - 热键 {idx+1}: ID='{hotkey_id}', 名称='{hotkey_name}'")
            else:
                logger.info("调试 - 没有可用的热键")
        except Exception as e:
            logger.error(f"调试热键异常: {e}")

async def get_vts_controller() -> VTubeStudioController:
    """获取VTube Studio控制器实例"""
    return VTubeStudioController(port=8002)

async def start(library: Optional[str] = None) -> None:
    """启动VTube Studio控制器测试
    
    Args:
        library: 要使用的库类型，可选值: "pyvts", "pytubestudio"，或None（自动选择）
    """
    try:
        # 创建控制器实例，指定要使用的库
        controller = VTubeStudioController(port=8002, library=library)
        
        # 显示使用的库信息
        logger.info(f"使用库: {controller.library_type}")
        
        # 连接到VTube Studio
        connected = await controller.connect()
        if not connected:
            logger.error("连接VTube Studio失败")
            return
        
        # 获取模型列表
        models = await controller.get_models()
        logger.info(f"VTube Studio模型数: {len(models)} 个")
        
        # 获取当前模型
        current_model = await controller.get_current_model()
        if current_model:
            logger.info(f"当前模型: {current_model.get('modelName', 'unknown')}")
        else:
            logger.warning("未获取到当前模型信息")
        
        # 获取可用热键列表
        hotkeys = await controller.get_hotkeys()
        logger.info(f"可用热键数: {len(hotkeys)} 个")
        
        # 关闭所有表情
        await controller.close_all_expression()
        
        # 尝试激活一个热键（如果使用pyvts库，使用不同的热键ID格式）
        if controller.library_type == "pyvts":
            # pyvts库测试热键
            if hotkeys:
                # 尝试使用第一个热键
                test_hotkey = hotkeys[0]
                hotkey_id = test_hotkey.get('hotkeyID', '')
                hotkey_name = test_hotkey.get('name', '')
                
                if hotkey_id or hotkey_name:
                    # 使用存在的热键ID或名称
                    hotkey_to_activate = hotkey_id or hotkey_name
                    logger.info(f"尝试使用pyvts激活热键 '{hotkey_to_activate}'")
                    result = await controller.activation_hotkey(hotkey_to_activate)
                    logger.info(f"热键激活结果: {result}")
                else:
                    # 如果获取不到热键，使用一个常见的热键名称作为测试
                    logger.info(f"尝试使用pyvts激活热键 '表情1'")
                    result = await controller.activation_hotkey("表情1")
                    logger.info(f"热键激活结果: {result}")
            else:
                logger.warning("没有可用的热键")
        else:
            # PyTubeStudio库测试热键
            logger.info(f"尝试使用PyTubeStudio激活热键 '5'")
            result = await controller.activation_hotkey(5)
            logger.info(f"热键激活结果: {result}")
        
        # 获取可用表情列表
        expressions = await controller.get_expressions()
        logger.info(f"VTube Studio表情数: {len(expressions)} 个")
        
        # 如果有可用表情，尝试设置表情
        if expressions:
            # 动态选择测试表情索引，优先选择有效的表情
            expression_index = 0
            if len(expressions) > 4 and len(expressions) > expression_index:
                # 如果有多个表情，使用有效的索引
                expression_index = min(4, len(expressions) - 1)  # 确保索引不越界
            
            # 从多个可能的索引中尝试找到有效的表情
            max_attempts = min(3, len(expressions))  # 最多尝试3个不同的表情
            expression_activated = False
            
            for attempt in range(max_attempts):
                idx = (expression_index + attempt) % len(expressions)  # 循环尝试不同的索引
                test_expression = expressions[idx]
                
                # 尝试从不同的字段获取表情名称
                expression_name = None
                if 'name' in test_expression and test_expression['name']:
                    expression_name = test_expression['name']
                elif 'file' in test_expression and test_expression['file']:
                    expression_name = test_expression['file']
                
                if expression_name:
                    # 移除扩展名（如果有）
                    if expression_name.endswith('.exp3.json'):
                        base_name = expression_name[:-12]  # 移除.exp3.json扩展名
                    else:
                        base_name = expression_name
                    
                    logger.info(f"尝试激活表情 ({idx+1}/{len(expressions)}): {base_name}")
                    result = await controller.set_expression(expression_name, True, fade_time=0.5)
                    
                    if result:
                        logger.info(f"表情激活成功: {base_name}")
                        expression_activated = True
                        
                        # 延迟2秒后关闭表情
                        await asyncio.sleep(2)
                        await controller.set_expression(expression_name, False, fade_time=0.5)
                        break
                    else:
                        logger.warning(f"表情激活失败: {base_name}")
            
            if not expression_activated:
                logger.warning("尝试多个表情后均激活失败")
        else:
            logger.warning("当前模型中没有可用的表情")
        
        # 断开连接
        await controller.disconnect()
        logger.info("测试成功完成")
    except Exception as e:
        logger.error(f"VTube Studio测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(start())