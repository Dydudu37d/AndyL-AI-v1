import asyncio
import logging
from pyvts.vts import vts

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_vts_connection")

async def simple_connection():
    """简单测试VTube Studio连接功能"""
    try:
        # 初始化VTS客户端
        logger.info("初始化VTS客户端...")
        plugin_info = {
            "plugin_name": "AI_VTuber_Plugin",
            "developer": "AndyL",
            "authentication_token_path": "./pyvts_token.txt"
        }
        vts_client = vts(plugin_info=plugin_info,vts_api_info="ws://0.0.0.0:8002")
        
        # 连接到VTS
        logger.info("连接到VTube Studio...")
        await vts_client.connect()
        logger.info("连接成功")
        
        # 获取连接状态（不是async方法，直接调用）
        connection_status = vts_client.get_connection_status()
        logger.info(f"连接状态: {connection_status}")
        
        # 查看VTube Studio状态（不需要认证的API）
        logger.info("请求VTube Studio状态...")
        state_request = {
            "api_name": "APIStateRequest",
            "api_version": "1.0",
            "request_id": "",
            "data": {}
        }
        state_response = await vts_client.request(state_request)
        logger.info(f"状态响应: {state_response}")
        
        # 断开连接
        logger.info("断开与VTube Studio的连接...")
        await vts_client.close()
        logger.info("测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_connection())