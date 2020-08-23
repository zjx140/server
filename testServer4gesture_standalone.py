# !--*-- coding: utf-8 --*--
from myConfig import *
import socket
import web
import os
from threading import Thread
import asyncio
import websockets

# 仅用于测试gesture_standalone，做接收手势识别结果的服务器用
# ip='127.0.0.1', port=8888, 服务器监听IP地址和端口号
# BUFSIZE = 20, Socket缓冲区大小
# closeWhenClientClosed=True，Socket服务器端在客户端关闭时是否随之关闭
# 即使closeWhenClientClosed=False，收到客户端CloseSever指令（连续按两次ESC键）时，Socket服务器也会关闭
def socketServer(ip='127.0.0.1', port=8888, BUFSIZE = 20, closeWhenClientClosed=True):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 重用IP和端口号
    s.bind((ip,port))
    s.listen(1)
    print('Socket服务器端启动，开始侦听手势识别结果......')

    while True:
        conn, addr = s.accept()
        print('接到来自%s的连接' % addr[0])
        while True:
            try:
                gesture = conn.recv(BUFSIZE)
                gesture=gesture.decode('utf-8')
            except:
                if not closeWhenClientClosed:
                    s.listen(1)
                    print('Socket服务器端重新开始侦听手势识别结果......')
                else:
                    print('客户端%s连接意外断开' % addr[0])
                break
            if len(gesture) > 0:
                if gesture=='CloseClient':
                    print('客户端%s通信结束，连接关闭。' % addr[0])
                    if not closeWhenClientClosed:
                        s.listen(1)
                        print('Socket服务器端重新开始侦听手势识别结果......')
                    break
                elif gesture=='CloseServer':
                    print('收到客户端%s发来的终止服务器程序指令，服务端程序终止。' % addr[0])
                    conn.close()
                    s.close()
                    return
                else:
                    print('通过Socket收到手势识别结果：',gesture)
        conn.close()
        if closeWhenClientClosed: break
    s.close()
    return 0

# RESTful接口定义
urls = ('/gesture', 'RestAPI')
RestAPP = web.application(urls, globals())
class RestAPI:
    def GET(self):
        gesture= web.input()
        if gesture['gesture'] == 'CloseClient':
            print('客户端已经关闭。')
        elif gesture['gesture']=='CloseServer':
            print('收到客户端发来的终止服务器程序指令，服务端程序终止。')
            os._exit(0) # 结束主进程
        else:
            print('接收到REST接口手势识别结果：',gesture['gesture'])
        return gesture

# WebSocketServer
# websocket和path是该函数被回调时自动传过来的，不需要自己传
async def wsServer(websocket, path):
    print('WebSocketServer收到来自',websocket.remote_address,'的连接请求')
    try:
        while True:
            gesture = await websocket.recv()
            gesture = gesture
            if len(gesture) > 0:
                if gesture=='CloseClient':
                    print('客户端通信结束，WebSocket服务器等候新的连接...')
                elif gesture=='CloseServer':
                    print('收到客户端发来的终止服务器程序指令，服务端程序终止。')
                    exit(0)
                else:
                    print('收到手势识别结果：',gesture)
    except Exception as e:
        if not '1000' in str(e):
            print('客户端已断开连接【',str(e),'】，WebSocket服务器等候新的连接...')

if __name__ == '__main__':
    # 仅用于测试gesture_standalone，做接收手势识别结果的服务器用
    if GestureSendMode=='SOCKET':
        print('启动SOCKET接口，准备接收手势识别结果......')
        socketServer(ip=SendGestureToServerIP, port=SendGestureToServerPort, BUFSIZE=20,closeWhenClientClosed=False)
    elif GestureSendMode=='REST':
        print('启动REST接口，准备接收手势识别结果（按回车键退出）......')
        tRestServer = Thread(target=RestAPP.run,daemon=True) # daemon=True,线程会随着主线程退出
        tRestServer.start()
        key=input()
        print('已退出REST服务端程序。')
    elif GestureSendMode == 'WEBSOCKET':
        print('启动WebSocket接口，准备接收手势识别结果......')
        start_server = websockets.serve(wsServer, SendGestureToWebSocketServerIP, SendGestureToWebSocketServerPort)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    else:
        print('没有设定启动哪种接口方式，未启动测试服务器。')
