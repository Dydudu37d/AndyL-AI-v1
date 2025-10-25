#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试OBS WebSocket API响应结构的脚本
打印原始响应数据，帮助了解OBS返回的实际数据格式
"""

import os
import sys
import logging
import time
from obs import OBSController

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('obs_debug.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数，连接OBS并打印原始响应数据"""
    try:
        # 创建OBS控制器实例
        obs_controller = OBSController()
        
        # 测试连接状态
        if not obs_controller.is_connected:
            logger.info("尝试连接到OBS WebSocket...")
            if not obs_controller.connect():
                logger.error("无法连接到OBS WebSocket")
                return
        
        logger.info("已成功连接到OBS WebSocket")
        
        # 获取当前场景名称
        scene_list = obs_controller.get_scene_list()
        if scene_list and isinstance(scene_list, dict):
            current_scene = scene_list.get('current_scene', '')
            logger.info(f"当前场景: {current_scene}")
        
        # 直接访问client对象来获取原始响应
        # 注意：这是调试代码，直接访问内部属性
        if hasattr(obs_controller, 'client'):
            client = obs_controller.client
            
            # 打印client对象的类型和可用方法
            logger.debug(f"client对象类型: {type(client)}")
            logger.debug(f"client对象可用方法: {[method for method in dir(client) if not method.startswith('_')]}")
            
            try:
                # 尝试获取场景项目列表的原始响应
                logger.info("尝试获取原始场景项目列表...")
                if current_scene:
                    # 使用位置参数name调用get_scene_item_list
                    raw_response = client.get_scene_item_list(name=current_scene)
                    logger.info(f"原始响应类型: {type(raw_response)}")
                    
                    # 打印原始响应的详细信息
                    logger.info("原始响应详细信息:")
                    if hasattr(raw_response, '__dict__'):
                        logger.info(f"响应属性: {raw_response.__dict__}")
                    elif isinstance(raw_response, dict):
                        logger.info(f"响应字典: {raw_response}")
                    else:
                        logger.info(f"响应内容: {raw_response}")
                    
                    # 尝试直接访问scene_items属性或键
                    if hasattr(raw_response, 'scene_items'):
                        scene_items = raw_response.scene_items
                        logger.info(f"scene_items类型: {type(scene_items)}")
                        logger.info(f"scene_items数量: {len(scene_items) if hasattr(scene_items, '__len__') else '未知'}")
                        
                        # 打印第一个scene item的详细信息
                        if scene_items and hasattr(scene_items, '__getitem__') and len(scene_items) > 0:
                            first_item = scene_items[0]
                            logger.info(f"第一个场景项类型: {type(first_item)}")
                            if hasattr(first_item, '__dict__'):
                                logger.info(f"第一个场景项属性: {first_item.__dict__}")
                            elif isinstance(first_item, dict):
                                logger.info(f"第一个场景项字典: {first_item}")
                    elif isinstance(raw_response, dict) and 'scene_items' in raw_response:
                        scene_items = raw_response['scene_items']
                        logger.info(f"scene_items类型: {type(scene_items)}")
                        logger.info(f"scene_items数量: {len(scene_items)}")
            except Exception as e:
                logger.error(f"获取原始场景项目列表时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n=== 调试信息已保存到 obs_debug.log ===")
        print("请查看日志文件以了解OBS WebSocket返回的详细数据结构")
        
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 断开连接
        if 'obs_controller' in locals() and obs_controller.is_connected:
            obs_controller.disconnect()
            logger.info("已断开与OBS的连接")


if __name__ == "__main__":
    main()