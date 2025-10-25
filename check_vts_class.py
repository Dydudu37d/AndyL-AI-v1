import pyvts
from pyvts import vts  # 尝试直接导入vts类
import inspect

print('pyvts版本:', pyvts.__version__)
print('vts类是否存在:', hasattr(pyvts, 'vts'))
print('vts类类型:', type(pyvts.vts))

# 查看vts类的所有属性和方法
print('\nvts类的属性和方法:', dir(pyvts.vts))

# 尝试创建vts类的实例
try:
    # 查看vts类的构造函数签名
    print('\nvts类构造函数签名:')
    constructor_sig = inspect.signature(pyvts.vts.__init__)
    print(constructor_sig)
    
    # 创建实例
    print('\n尝试创建vts实例...')
    client = pyvts.vts(port=8002)
    print('成功创建vts实例')
    
    # 查看实例的方法
    instance_methods = [attr for attr in dir(client) if callable(getattr(client, attr)) and not attr.startswith('__')]
    print('\nvts实例的方法:', instance_methods)
    
    # 特别检查request方法
    if hasattr(client, 'request'):
        print('\nrequest方法存在')
        request_method = client.request
        print('request方法签名:')
        request_sig = inspect.signature(request_method)
        print(request_sig)
        
        # 检查方法的文档字符串
        if request_method.__doc__:
            print('\nrequest方法文档:')
            print(request_method.__doc__)
    
    # 检查connect方法
    if hasattr(client, 'connect'):
        print('\nconnect方法存在')
        connect_method = client.connect
        print('connect方法签名:')
        connect_sig = inspect.signature(connect_method)
        print(connect_sig)
    
    # 检查authenticate方法
    if hasattr(client, 'authenticate'):
        print('\nauthenticate方法存在')
        auth_method = client.authenticate
        print('authenticate方法签名:')
        auth_sig = inspect.signature(auth_method)
        print(auth_sig)
        
except Exception as e:
    print(f'创建vts实例或检查方法时出错: {e}')

# 检查VTSRequest类
if hasattr(pyvts, 'VTSRequest'):
    print('\n\n检查VTSRequest类...')
    print('VTSRequest类型:', type(pyvts.VTSRequest))
    print('VTSRequest属性和方法:', dir(pyvts.VTSRequest))
    
    # 查看VTSRequest中的一些可能的方法
    if hasattr(pyvts.VTSRequest, 'authentication'):
        print('\nVTSRequest.authentication:')
        auth_method = pyvts.VTSRequest.authentication
        print(f'类型: {type(auth_method)}')
        if inspect.ismethod(auth_method) or inspect.isfunction(auth_method):
            try:
                sig = inspect.signature(auth_method)
                print(f'签名: {sig}')
            except Exception as e:
                print(f'无法获取签名: {e}')
    
    # 查看其他可能的API请求方法
    for method_name in dir(pyvts.VTSRequest):
        if not method_name.startswith('__') and callable(getattr(pyvts.VTSRequest, method_name)):
            print(f'\nVTSRequest.{method_name}:')
            method = getattr(pyvts.VTSRequest, method_name)
            if inspect.ismethod(method) or inspect.isfunction(method):
                try:
                    sig = inspect.signature(method)
                    print(f'类型: {type(method)}')
                    print(f'签名: {sig}')
                except Exception as e:
                    print(f'无法获取签名: {e}')