#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式OBS文本源管理工具
这个工具可以帮助用户查找、创建和修改OBS中的文本源，解决"未检测到文本源"的问题。
"""
import time
import sys
from obs import obs_controller

class OBSTextTool:
    def __init__(self):
        self.controller = obs_controller
        self.connected = False
        
    def connect(self):
        """连接到OBS WebSocket服务器"""
        if self.controller.is_connected:
            print("✅ OBS WebSocket连接成功!")
            self.connected = True
        else:
            print("❌ OBS WebSocket连接失败!")
            print("请确认：")
            print("1. OBS Studio正在运行")
            print("2. WebSocket服务器已在OBS设置中启用")
            print("3. 连接参数（主机、端口、密码）正确")
            self.connected = False
        return self.connected
    
    def list_scene_items(self):
        """列出当前场景中的所有来源"""
        if not self.connected:
            return []
        
        scene_items = self.controller.get_scene_items()
        if not scene_items:
            print("❌ 无法获取场景来源")
            return []
        
        print(f"\n当前场景中的来源 ({len(scene_items)}个):")
        text_sources = []
        
        for i, item in enumerate(scene_items, 1):
            # 尝试获取来源信息
            item_name = '未知名称'
            item_type = '未知类型'
            
            if isinstance(item, dict):
                item_name = item.get('source_name', '未知名称')
                item_type = item.get('source_type', '未知类型')
            elif hasattr(item, 'source_name'):
                item_name = item.source_name
                if hasattr(item, 'source_type'):
                    item_type = item.source_type
            
            print(f"{i}. {item_name} (类型: {item_type})")
            
            # 收集文本源（根据类型判断）
            if 'text' in item_type.lower() or 'text_gdiplus' in item_type.lower():
                text_sources.append(item_name)
        
        if text_sources:
            print(f"\n检测到的文本源 ({len(text_sources)}个):")
            for i, text_source in enumerate(text_sources, 1):
                print(f"{i}. {text_source}")
        else:
            print("\n提示：当前场景中未检测到文本源。")
            print("可以使用此工具创建新的文本源测试，或在OBS中手动创建。")
        
        return text_sources
    
    def create_text_source(self, source_name="测试文本源", initial_text="这是一个测试文本"):
        """创建一个新的文本源（注意：此功能需要OBS支持，可能需要额外配置）"""
        print(f"\n尝试创建新文本源: {source_name}")
        print("注意：直接通过WebSocket API创建文本源需要OBS支持高级功能。")
        print("推荐方法：请在OBS中手动创建文本源，步骤如下：")
        print("1. 在OBS主界面点击'+'按钮添加来源")
        print("2. 选择'文本 (GDI+)'或'文本 (FreeType 2)'")
        print("3. 命名文本源（例如：'测试文本'）")
        print("4. 点击确定并设置初始文本")
        print("5. 完成后返回此工具继续操作")
        
        # 尝试通过文本修改方式间接创建（仅在部分OBS版本可能有效）
        try:
            print(f"\n尝试通过修改方式创建文本源...")
            result = self.controller.set_text_source_content(source_name, initial_text)
            if result:
                print(f"✅ 文本源创建/修改成功！")
                return True
            else:
                print(f"提示：如果文本源不存在，某些OBS版本可能需要先在OBS界面手动创建。")
                return False
        except Exception as e:
            print(f"❌ 创建文本源时出错: {e}")
            return False
    
    def modify_text_source(self, source_name, new_text):
        """修改指定文本源的内容"""
        if not self.connected:
            return False
        
        print(f"\n尝试修改文本源 '{source_name}' 的内容为:")
        print(f"{new_text}")
        
        success = self.controller.set_text_source_content(source_name, new_text)
        
        if success:
            print(f"✅ 文本源内容修改成功！")
            # 等待1秒让用户看到效果
            time.sleep(1)
            return True
        else:
            print(f"❌ 文本源内容修改失败！")
            print("可能的原因：")
            print("- 文本源名称不正确")
            print("- OBS连接中断")
            print("- 文本源类型不支持直接修改")
            return False
    
    def interactive_mode(self):
        """交互式模式，引导用户操作"""
        print("=== OBS文本源交互式管理工具 ===")
        
        # 连接到OBS
        if not self.connect():
            return
        
        while True:
            print("\n请选择操作:")
            print("1. 列出当前场景中的所有来源和文本源")
            print("2. 修改现有文本源的内容")
            print("3. 尝试创建新的文本源（或测试指定名称的文本源）")
            print("4. 退出工具")
            
            choice = input("请输入选项编号 (1-4): ").strip()
            
            if choice == '1':
                self.list_scene_items()
            elif choice == '2':
                text_sources = self.list_scene_items()
                if not text_sources:
                    print("\n当前没有可修改的文本源。")
                    source_name = input("请输入文本源名称（可尝试手动输入）: ").strip()
                else:
                    print("\n请选择要修改的文本源编号 (1-{}): ".format(len(text_sources)))
                    try:
                        index = int(input().strip()) - 1
                        if 0 <= index < len(text_sources):
                            source_name = text_sources[index]
                        else:
                            print("❌ 无效的选择。")
                            continue
                    except ValueError:
                        print("❌ 请输入有效的数字。")
                        continue
                
                new_text = input("请输入新的文本内容: ").strip()
                self.modify_text_source(source_name, new_text)
            elif choice == '3':
                source_name = input("请输入新文本源的名称（或测试名称）: ").strip()
                if not source_name:
                    source_name = "测试文本源"
                initial_text = input("请输入初始文本内容: ").strip()
                if not initial_text:
                    initial_text = "这是一个测试文本"
                self.create_text_source(source_name, initial_text)
            elif choice == '4':
                print("\n感谢使用OBS文本源管理工具！")
                break
            else:
                print("❌ 无效的选项，请重新输入。")
    
    def text(self, source_name, new_text):
        """修改指定文本源的内容"""
        if not self.connected:
            return False
        
        print(f"\n尝试修改文本源 '{source_name}' 的内容为:")
        print(f"{new_text}")
        
        success = self.controller.set_text_source_content(source_name, new_text)
        

if __name__ == "__main__":
    tool = OBSTextTool()
    tool.interactive_mode()