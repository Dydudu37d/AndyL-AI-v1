#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OBS控制器模块
使用obsws-python库与OBS Studio WebSocket 5.x通信
"""
import logging
import time
from obswebsocket import *
import sys
import importlib
import subprocess
import os
import socket

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OBSController')

# 尝试导入obsws-python库
import obsws_python
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
host = local_ip
port = 4455
password = '5he8ccylTZTWngjq'
try:
    from obsws_python import ReqClient, reqs
    from obsws_python.error import OBSSDKError
    print("✅ 成功导入obsws-python库，使用正确的API")
    OBS_LIBRARY_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入obsws-python库失败: {e}")
    logger.warning("未找到obsws-python库，尝试安装: pip install obsws-python")
    OBS_LIBRARY_AVAILABLE = False
    # 定义模拟类以避免代码错误
    class OBSSDKError(Exception):
        pass
    class reqs:
        class GetSceneList:
            pass
        class SetCurrentProgramScene:
            def __init__(self, **kwargs):
                pass
        class StartRecord:
            pass
        class StopRecord:
            pass
        class StartStream:
            pass
        class StopStream:
            pass
        class GetStreamStatus:
            pass
        class GetRecordStatus:
            pass
    class ReqClient:
        def __init__(self, **kwargs):
            pass
        def connect(self):
            raise ImportError("obsws-python库未安装")
        def call(self, req):
            raise ImportError("obsws-python库未安装")
        def disconnect(self):
            pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OBSController')


class OBSController:
    """OBS控制器类 - 提供与OBS Studio交互的各种方法"""
    
    def __init__(self, host='192.168.0.186', port=4455, password='5he8ccylTZTWngjq', auto_connect=True):
        """
        初始化OBS控制器
        
        Args:
            host (str): OBS WebSocket服务器地址
            port (int): OBS WebSocket服务器端口
            password (str): OBS WebSocket服务器密码
            auto_connect (bool): 是否自动连接
        """
        self.host = host
        self.port = port
        self.password = password
        self.client = None
        self.is_connected = False
        
        if auto_connect:
            self.connect()
    
    def connect(self):
        """连接到OBS WebSocket服务器"""
        if not OBS_LIBRARY_AVAILABLE:
            logger.error("obsws-python库未安装，无法连接OBS")
            return False
            
        try:
            logger.info(f"尝试连接到OBS WebSocket服务器: {self.host}:{self.port}")
            # 使用正确的ReqClient类和参数
            self.client = ReqClient(host=self.host, port=self.port, password=self.password)
            # 注意：在obsws-python中，connect方法不需要显式调用，初始化时就会连接
            self.is_connected = True
            logger.info("OBS WebSocket连接成功!")
            return True
        except Exception as e:
            logger.error(f"OBS WebSocket连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开与OBS WebSocket服务器的连接"""
        try:
            if self.client and self.is_connected:
                self.client.disconnect()
                self.is_connected = False
                logger.info("OBS WebSocket连接已断开")
            return True
        except Exception as e:
            logger.error(f"断开OBS WebSocket连接时出错: {e}")
            return False
    
    def reconnect(self, max_attempts=3, delay=2):
        """重新连接到OBS WebSocket服务器"""
        self.disconnect()
        
        for attempt in range(max_attempts):
            logger.info(f"重新连接尝试 {attempt+1}/{max_attempts}")
            if self.connect():
                return True
            logger.info(f"等待{delay}秒后重试...")
            time.sleep(delay)
        
        logger.error(f"达到最大重连次数({max_attempts})，连接失败")
        return False
    
    def get_scene_list(self):
        """获取场景列表"""
        try:
            if not self._check_connection():
                return None
            
            response = self.client.get_scene_list()
            
            if hasattr(response, 'scenes'):
                scenes = response.scenes
                current_program_scene_name = response.current_program_scene_name
            else:
                scenes = response.get('scenes', [])
                current_program_scene_name = response.get('current_program_scene_name', '')
            
            # 格式化场景列表，确保返回一致的格式
            formatted_scenes = []
            for scene in scenes:
                formatted_scene = {}
                
                if hasattr(scene, 'scene_name'):
                    formatted_scene['scene_name'] = scene.scene_name
                elif hasattr(scene, 'sceneName'):
                    formatted_scene['scene_name'] = scene.sceneName
                elif isinstance(scene, dict):
                    formatted_scene['scene_name'] = scene.get('scene_name', scene.get('sceneName', '未知场景'))
                else:
                    formatted_scene['scene_name'] = '未知场景'
                
                # 保留原始场景对象的其他属性
                if hasattr(scene, '__dict__'):
                    formatted_scene.update(scene.__dict__)
                elif isinstance(scene, dict):
                    formatted_scene.update(scene)
                
                formatted_scenes.append(formatted_scene)
            
            scene_list = {
                'scenes': formatted_scenes,
                'current_scene': current_program_scene_name
            }
            
            logger.info(f"获取场景列表成功，共{len(formatted_scenes)}个场景，当前场景: {current_program_scene_name}")
            return scene_list
        except Exception as e:
            logger.error(f"获取场景列表时出错: {e}")
            return False
        
    
    def set_current_scene(self, scene_name):
        """切换到指定场景"""
        try:
            if not self._check_connection():
                return False
            
            logger.info(f"尝试切换到场景: {scene_name}")
            # 使用正确的reqs.SetCurrentProgramScene
            self.client.call(reqs.SetCurrentProgramScene(scene_name=scene_name))
            logger.info(f"场景切换成功: {scene_name}")
            return True
        except OBSSDKError as e:
            logger.error(f"切换场景时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"切换场景时出错: {e}")
            return False
    
    def start_recording(self):
        """开始录制"""
        try:
            if not self._check_connection():
                return False
            
            logger.info("尝试开始录制")
            # 使用正确的reqs.StartRecord
            self.client.call(reqs.StartRecord())
            logger.info("录制已开始")
            return True
        except OBSSDKError as e:
            logger.error(f"开始录制时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"开始录制时出错: {e}")
            return False
    
    def stop_recording(self):
        """停止录制"""
        try:
            if not self._check_connection():
                return False
            
            logger.info("尝试停止录制")
            # 使用正确的reqs.StopRecord
            self.client.call(reqs.StopRecord())
            logger.info("录制已停止")
            return True
        except OBSSDKError as e:
            logger.error(f"停止录制时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"停止录制时出错: {e}")
            return False
    
    def start_streaming(self):
        """开始直播"""
        try:
            if not self._check_connection():
                return False
            
            logger.info("尝试开始直播")
            # 使用正确的reqs.StartStream
            self.client.call(reqs.StartStream())
            logger.info("直播已开始")
            return True
        except OBSSDKError as e:
            logger.error(f"开始直播时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"开始直播时出错: {e}")
            return False
    
    def stop_streaming(self):
        """停止直播"""
        try:
            if not self._check_connection():
                return False
            
            logger.info("尝试停止直播")
            # 使用正确的reqs.StopStream
            self.client.call(reqs.StopStream())
            logger.info("直播已停止")
            return True
        except OBSSDKError as e:
            logger.error(f"停止直播时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"停止直播时出错: {e}")
            return False
    
    def get_stream_status(self):
        """获取直播状态"""
        try:
            if not self._check_connection():
                return None
            
            # 使用obsws-python库的特定方法get_stream_status
            response = self.client.get_stream_status()
            
            # 构建状态字典
            status = {
                'streaming': False,
                'recording': False,
                'preview_only': False
            }
            
            # 处理流状态
            status['streaming'] = getattr(response, 'output_active', False)
            
            # 获取录制状态
            try:
                record_response = self.client.get_record_status()
                status['recording'] = getattr(record_response, 'output_active', False)
            except Exception as e:
                logger.warning(f"获取录制状态时出错: {e}")
            
            logger.info(f"获取直播状态成功: {status}")
            return status
        except OBSSDKError as e:
            logger.error(f"获取直播状态时OBS SDK错误: {e}")
            return None
        except Exception as e:
            logger.error(f"获取直播状态时出错: {e}")
            return None
    
    def get_scene_items(self, scene_name=None):
        """获取指定场景或当前场景下的来源"""
        try:
            if not self._check_connection():
                return None
            
            # 如果没有指定场景名称，使用当前场景
            if scene_name is None:
                scene_list = self.get_scene_list()
                if scene_list:
                    scene_name = scene_list['current_scene']
                else:
                    logger.error("无法获取当前场景名称")
                    return None
                
            # 根据错误信息，get_scene_item_list需要一个名为'name'的位置参数
            logger.info(f"尝试获取场景 '{scene_name}' 下的来源")
            try:
                # 使用位置参数'name'
                response = self.client.get_scene_item_list(name=scene_name)
            except TypeError:
                try:
                    # 直接作为位置参数传递
                    logger.info(f"尝试直接作为位置参数传递场景名称")
                    response = self.client.get_scene_item_list(scene_name)
                except Exception as e:
                    logger.error(f"尝试获取场景来源失败: {e}")
                    return None
            
            # 处理响应数据，适配不同的响应格式
            scene_items = []
            
            # 检查响应类型并提取scene_items
            if hasattr(response, 'scene_items'):
                scene_items = response.scene_items
            elif hasattr(response, '__dict__'):
                # 尝试从对象的__dict__中获取scene_items
                scene_items = response.__dict__.get('scene_items', [])
            elif isinstance(response, dict):
                # 从字典中尝试不同的键名
                scene_items = response.get('scene_items', [])
                if not scene_items:
                    scene_items = response.get('sceneItems', [])
            
            # 如果scene_items还是空的，尝试将整个响应作为scene_items
            if not scene_items and isinstance(response, (list, tuple)):
                scene_items = response
                logger.info(f"将整个响应作为场景来源列表")
            
            # 格式化场景来源数据，确保返回一致的字典格式
            formatted_items = []
            for idx, item in enumerate(scene_items):
                formatted_item = {}
                
                # 尝试从不同的属性或键中提取来源名称
                # 增强兼容性，处理更多可能的数据结构
                source_name_found = False
                
                # 先检查是否有嵌套的source对象
                if hasattr(item, 'source') and item.source:
                    source_obj = item.source
                    if hasattr(source_obj, 'name'):
                        formatted_item['source_name'] = source_obj.name
                        source_name_found = True
                    elif isinstance(source_obj, dict):
                        formatted_item['source_name'] = source_obj.get('name', '未知名称')
                        source_name_found = True
                
                # 如果item是字典，尝试从更多可能的键中提取来源名称
                if not source_name_found and isinstance(item, dict):
                    # 检查是否有嵌套的source对象
                    if 'source' in item and item['source']:
                        source_obj = item['source']
                        if isinstance(source_obj, dict):
                            formatted_item['source_name'] = source_obj.get('name', '未知名称')
                            source_name_found = True
                    
                    # 尝试更多可能的键名
                    if not source_name_found:
                        possible_keys = ['source_name', 'sceneItemName', 'name', 'inputName', 'input_name']
                        for key in possible_keys:
                            if key in item and item[key]:
                                formatted_item['source_name'] = item[key]
                                source_name_found = True
                                break
                
                # 使用基本属性检查
                if not source_name_found:
                    if hasattr(item, 'source_name'):
                        formatted_item['source_name'] = item.source_name
                        source_name_found = True
                    elif hasattr(item, 'sceneItemName'):
                        formatted_item['source_name'] = item.sceneItemName
                        source_name_found = True
                    elif hasattr(item, 'name'):
                        formatted_item['source_name'] = item.name
                        source_name_found = True
                    elif hasattr(item, 'inputName'):
                        formatted_item['source_name'] = item.inputName
                        source_name_found = True
                    elif hasattr(item, 'input_name'):
                        formatted_item['source_name'] = item.input_name
                        source_name_found = True
                
                # 如果还是没找到，设置为未知名称并记录调试信息
                if not source_name_found:
                    formatted_item['source_name'] = '未知名称'
                    logger.debug(f"场景来源 {idx} 无法提取名称，原始数据: {type(item)} {item}")
                
                # 提取ID信息，增强兼容性
                scene_item_id_found = False
                
                # 尝试从嵌套结构中提取
                if isinstance(item, dict):
                    possible_id_keys = ['scene_item_id', 'sceneItemId', 'id', 'itemId']
                    for key in possible_id_keys:
                        if key in item and item[key]:
                            formatted_item['scene_item_id'] = item[key]
                            scene_item_id_found = True
                            break
                
                # 使用基本属性检查
                if not scene_item_id_found:
                    if hasattr(item, 'scene_item_id'):
                        formatted_item['scene_item_id'] = item.scene_item_id
                        scene_item_id_found = True
                    elif hasattr(item, 'sceneItemId'):
                        formatted_item['scene_item_id'] = item.sceneItemId
                        scene_item_id_found = True
                    elif hasattr(item, 'id'):
                        formatted_item['scene_item_id'] = item.id
                        scene_item_id_found = True
                
                if not scene_item_id_found:
                    formatted_item['scene_item_id'] = '未知ID'
                
                # 提取类型信息，增强兼容性
                source_type_found = False
                
                # 检查是否有嵌套的source对象
                if hasattr(item, 'source') and item.source:
                    source_obj = item.source
                    if hasattr(source_obj, 'type'):
                        formatted_item['source_type'] = source_obj.type
                        source_type_found = True
                    elif isinstance(source_obj, dict):
                        formatted_item['source_type'] = source_obj.get('type', '未知类型')
                        source_type_found = True
                
                # 如果item是字典，尝试从更多可能的键中提取类型
                if not source_type_found and isinstance(item, dict):
                    if 'source' in item and item['source']:
                        source_obj = item['source']
                        if isinstance(source_obj, dict):
                            formatted_item['source_type'] = source_obj.get('type', '未知类型')
                            source_type_found = True
                    
                    if not source_type_found:
                        possible_type_keys = ['source_type', 'sourceType', 'type']
                        for key in possible_type_keys:
                            if key in item and item[key]:
                                formatted_item['source_type'] = item[key]
                                source_type_found = True
                                break
                
                # 使用基本属性检查
                if not source_type_found:
                    if hasattr(item, 'source_type'):
                        formatted_item['source_type'] = item.source_type
                        source_type_found = True
                    elif hasattr(item, 'sourceType'):
                        formatted_item['source_type'] = item.sourceType
                        source_type_found = True
                    elif hasattr(item, 'type'):
                        formatted_item['source_type'] = item.type
                        source_type_found = True
                
                if not source_type_found:
                    formatted_item['source_type'] = '未知类型'
                
                formatted_items.append(formatted_item)
            
            logger.info(f"获取场景 '{scene_name}' 下的来源成功，共{len(formatted_items)}个来源")
            return formatted_items
        except OBSSDKError as e:
            logger.error(f"获取场景下的来源时OBS SDK错误: {e}")
            return None
        except Exception as e:
            logger.error(f"获取场景下的来源时出错: {e}")
            return None
    
    def _check_connection(self):
        """检查连接状态，如果断开则尝试重连"""
        if not self.is_connected:
            logger.warning("OBS WebSocket连接已断开，尝试重连...")
            return self.reconnect()
        return True
        
    def set_text_source_content(self, source_name, text_content):
        """修改OBS文本源的内容
        
        Args:
            source_name (str): 文本源的名称
            text_content (str): 新的文本内容
        
        Returns:
            bool: 操作是否成功
        """
        try:
            if not self._check_connection():
                return False
            
            logger.info(f"尝试修改文本源 '{source_name}' 的内容")
            
            # 使用obsws-python库的set_input_settings方法设置文本源内容
            # 对于不同类型的文本源，参数可能会有所不同
            # 这里主要处理text_gdiplus和text_gdiplus_v3类型的文本源
            
            # 准备要设置的输入设置
            settings = {
                "text": text_content
            }
            
            # 尝试不同的参数名称组合，适配不同版本的OBS WebSocket API
            try:
                # 尝试方法1: 使用inputName参数名
                response = self.client.set_input_settings(
                    inputName=source_name, 
                    settings=settings, 
                    overlay=True
                )
            except TypeError:
                try:
                    # 尝试方法2: 使用input_name参数名
                    response = self.client.set_input_settings(
                        input_name=source_name, 
                        settings=settings, 
                        overlay=True
                    )
                except TypeError:
                    try:
                        # 尝试方法3: 使用source_name参数名
                        response = self.client.set_input_settings(
                            source_name=source_name, 
                            settings=settings, 
                            overlay=True
                        )
                    except TypeError:
                        # 尝试方法4: 直接位置参数传递（不使用关键字参数）
                        logger.info(f"尝试直接传递位置参数")
                        response = self.client.set_input_settings(
                            source_name, 
                            settings, 
                            True
                        )
            
            logger.info(f"文本源 '{source_name}' 内容更新成功")
            return True
        except OBSSDKError as e:
            logger.error(f"修改文本源内容时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"修改文本源内容时出错: {e}")
            return False
            
    def set_scene_item_visible(self, scene_item_id, visible=True, scene_name=None):
        """根据ID设置场景中来源的可见性
        
        Args:
            scene_item_id (int): 场景来源的ID
            visible (bool): 设置来源是否可见
            scene_name (str, optional): 场景名称，如果不指定则使用当前场景
        
        Returns:
            bool: 操作是否成功
        """
        try:
            if not self._check_connection():
                return False
            
            # 如果没有指定场景名称，使用当前场景
            if scene_name is None:
                scene_list = self.get_scene_list()
                if scene_list:
                    scene_name = scene_list['current_scene']
                else:
                    logger.error("无法获取当前场景名称")
                    return None
            
            logger.info(f"尝试设置场景 '{scene_name}' 中ID为 {scene_item_id} 的来源可见性为 {visible}")
            
            # 尝试不同的参数名称组合，适配不同版本的OBS WebSocket API
            try:
                # 尝试方法1: 使用id参数名
                response = self.client.set_scene_item_enabled(
                    scene_name=scene_name, 
                    id=scene_item_id, 
                    scene_item_enabled=visible
                )
            except TypeError:
                try:
                    # 尝试方法2: 使用item_id参数名
                    response = self.client.set_scene_item_enabled(
                        scene_name=scene_name, 
                        item_id=scene_item_id, 
                        scene_item_enabled=visible
                    )
                except TypeError:
                    try:
                        # 尝试方法3: 使用sceneItemId参数名
                        response = self.client.set_scene_item_enabled(
                            scene_name=scene_name, 
                            sceneItemId=scene_item_id, 
                            scene_item_enabled=visible
                        )
                    except TypeError:
                        # 尝试方法4: 直接位置参数传递（不使用关键字参数）
                        logger.info(f"尝试直接传递位置参数")
                        response = self.client.set_scene_item_enabled(
                            scene_name, 
                            scene_item_id, 
                            visible
                        )
            
            logger.info(f"来源ID {scene_item_id} 的可见性设置成功")
            return True
        except OBSSDKError as e:
            logger.error(f"设置来源可见性时OBS SDK错误: {e}")
            return False
        except Exception as e:
            logger.error(f"设置来源可见性时出错: {e}")
            return False
            
    def toggle_scene_item_visibility(self, scene_item_id, scene_name=None):
        """根据ID切换场景中来源的可见性
        
        Args:
            scene_item_id (int): 场景来源的ID
            scene_name (str, optional): 场景名称，如果不指定则使用当前场景
        
        Returns:
            bool or None: 操作是否成功，成功时返回新的可见性状态
        """
        try:
            if not self._check_connection():
                return False
            
            # 首先获取来源的当前可见性状态
            scene_items = self.get_scene_items(scene_name)
            if scene_items:
                for item in scene_items:
                    if item.get('scene_item_id') == scene_item_id:
                        # 获取变换信息以确定可见性
                        try:
                            # 尝试获取来源的变换信息，这通常包含可见性状态
                            transform_response = self.client.get_scene_item_transform(
                                scene_name=scene_name, 
                                scene_item_id=scene_item_id
                            )
                            
                            # 检查当前可见性状态（通常在transform信息中）
                            # 注意：不同版本的OBS API可能返回不同格式的数据
                            current_visible = True
                            if hasattr(transform_response, 'scene_item_transform'):
                                transform = transform_response.scene_item_transform
                                # 有些OBS API版本可能将可见性存储在不同的位置
                                if hasattr(transform, 'visible'):
                                    current_visible = transform.visible
                                elif hasattr(transform, 'scene_item_enabled'):
                                    current_visible = transform.scene_item_enabled
                        except:
                            # 如果无法获取当前可见性，默认切换为隐藏
                            current_visible = True
                            
                        # 设置新的可见性状态（与当前相反）
                        new_visible = not current_visible
                        success = self.set_scene_item_visible(scene_item_id, new_visible, scene_name)
                        
                        if success:
                            logger.info(f"成功切换来源ID {scene_item_id} 的可见性，新状态: {new_visible}")
                            return new_visible
                        else:
                            return None
                
            logger.error(f"未找到ID为 {scene_item_id} 的来源")
            return None
        except Exception as e:
            logger.error(f"切换来源可见性时出错: {e}")
            return None



# 创建全局OBS控制器实例（使用用户提供的连接信息）
local_ip = get_local_ip()
print(f"本机内网IP: {local_ip}")

obs_controller = OBSController(
    host=local_ip,
    port=4455,
    password='5he8ccylTZTWngjq'
)

if __name__ == "__main__":
    """测试OBS控制器功能"""
    # 测试连接状态
    if obs_controller.is_connected:
        print("OBS WebSocket连接成功!")
        
        # 测试获取场景列表
        scene_list = obs_controller.get_scene_list()
        if scene_list:
            print(f"\n可用场景列表:")
            for i, scene in enumerate(scene_list['scenes'], 1):
                current_marker = " * " if scene == scene_list['current_scene'] else "   "
                print(f"{i}.{current_marker}{scene}")
        
        # 测试获取直播状态
        stream_status = obs_controller.get_stream_status()
        if stream_status:
            print(f"\n当前状态:")
            print(f"- 直播中: {'是' if stream_status['streaming'] else '否'}")
            print(f"- 录制中: {'是' if stream_status['recording'] else '否'}")
            
        # 测试获取场景来源
        scene_items = obs_controller.get_scene_items()
        if scene_items:
            print(f"\n当前场景下的来源 ({len(scene_items)}个):")
            text_sources = []
            
            for i, item in enumerate(scene_items, 1):
                # 尝试获取来源名称，适配不同格式的响应
                item_name = '未知名称'
                item_type = '未知类型'
                
                if hasattr(item, 'source_name'):
                    item_name = item.source_name
                elif isinstance(item, dict):
                    item_name = item.get('source_name', '未知名称')
                    item_type = item.get('source_type', '未知类型')
                
                print(f"{i}.   {item_name} ({item_type})")
                
                # 收集文本源
                if isinstance(item, dict) and 'text_gdiplus' in item_type.lower():
                    text_sources.append(item_name)
        
        # 测试修改文本源内容
        if text_sources:
            print(f"\n检测到文本源: {text_sources}")
            test_text_source = text_sources[0]  # 使用第一个文本源进行测试
            new_text = "这是测试文本内容，OBS文本源已更新！"
            
            print(f"\n尝试修改文本源 '{test_text_source}' 的内容为:\n{new_text}")
            success = obs_controller.set_text_source_content(test_text_source, new_text)
            
            if success:
                print(f"✅ 文本源内容修改成功！")
                # 等待1秒让用户看到效果
                time.sleep(1)
                # 恢复原来的内容（可选）
                # obs_controller.set_text_source_content(test_text_source, "原来的文本内容")
            else:
                print(f"❌ 文本源内容修改失败！")
        else:
            print("\n提示：当前场景中未检测到文本源。")
            print("要测试修改文本功能，请先在OBS中添加文本源，然后运行此脚本。")
            print("或者，您可以在代码中手动指定文本源名称：")
            print("示例: obs_controller.set_text_source_content('您的文本源名称', '新的文本内容')")
    else:
        print("OBS WebSocket连接失败!")
        if not OBS_LIBRARY_AVAILABLE:
            print("请先安装obsws-python库: pip install obsws-python")
