from myConfig import *
from myWebsockets import client_connect_send
import asyncio

if __name__ == '__main__':
    print('启动Websock客户端连接5002并发送startCapturingVideo指令...')
    cmd='startCapturingVideo'
    # cmd = 'stopCapturingVideo'
    asyncio.get_event_loop().run_until_complete(client_connect_send(ip=VideoCMDIP, port=str(VideoCMPPort),cmd=cmd))
    print('WebSocket5002 Start客户端程序已退出')
