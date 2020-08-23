# !--*-- coding: utf-8 --*--
from myConfig import *
import asyncio
import websockets
import functools
import time
from sklearn.externals import joblib
from threading import Lock, Thread
from myFrameProcessing import frameProcess
from myFaceDetection import faceDection
from myCommonModules import *

if AsGestureRestServer:
    import web

if HandShapeDNNPath is not None:
    from myHandDNN import handDNN

# 全局变量，用于记录WebSocketServer两个连接
ws5002=None
ws5004=None
stopCapturingVideo=False # 停止发送视频
#codec = av.CodecContext.create('h264', 'r')# 视频解码器

if AsGestureRestServer:
    # 全局变量，用于REST返回,web.前缀是必须的，否则REST接口返回的都是初值
    web.gGESTURE = 'invalid'# 手势名称
    web.gConfidence = 1.00 # 置信度

    # 全局变量互斥锁
    lock = Lock()
    # RESTful接口定义
    urls = ('/gesture', 'RestAPI')
    RestAPP = myRestAPP(urls, globals())

    class RestAPI:
        def GET(self):
            print('接收到REST接口手势识别结果请求，返回：',end='')
            ret = lock.acquire(blocking = False)
            if ret:
                gestureJSON=gesture2JSON(web.gGESTURE,web.gConfidence)
                lock.release()
            print(gestureJSON)
            return gestureJSON


# 服务器端代码================================================
# 检测客户端权限，用户名密码通过才能退出循环，未使用
async def check_permit(websocket):
    while True:
        recv_str = await websocket.recv()
        cred_dict = recv_str.split(":")
        if cred_dict[0] == "admin" and cred_dict[1] == "123456":
            response_str = "congratulation, you have connect with server\r\nnow, you can do something else"
            await websocket.send(response_str)
            return True
        else:
            response_str = "sorry, the username or password is wrong, please submit again"
            await websocket.send(response_str)

# 接收客户端消息并处理，这里只是简单把客户端发来的返回回去
async def recv_msg(websocket, port=None):
    try:
        while True:
            recv_text = await websocket.recv()
            response_text = f"WebSocket服务器端收到: {recv_text}"
            await websocket.send(response_text)
    except Exception as e:
        if websocket.closed:
            print('WebSocketServer',port,'端口来自',websocket.remote_address,'的连接已关闭')
        else:
            print('WebSocketServer',port,'端口消息处理程序出错已退出，错误原因：',str(e))

# 接收5002客户端消息，通过向5004客户端传数据，模拟机器人视频数据下发就在这里
async def recv_send(websocket5002,websocket5004,port=None):
    global stopCapturingVideo
    #BUFFER = 1000000 # Websocket缓冲区大小，超过1048000，客户端缓冲区会溢出
    try:
        while True:
            recv_text = await websocket5002.recv()
            cmd_text = json.loads(recv_text)
            print('WebSocketServer', port, '端口收到来自', websocket5002.remote_address, '消息：', recv_text, '，解析出指令',
                  cmd_text['cmd'])
            if cmd_text['cmd'] == 'startCapturingVideo':
                if not websocket5004 is None:
                    stopCapturingVideo = False
                    sentPackets = 0 # 发送数据包数量
                    while not stopCapturingVideo:
                        # f=open(r'../video/test2.264', 'rb')
                        # data=f.read(BUFFER)
                        cap = myVideoCapture(cameraID=CameraID)
                        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                        print('FrameWidth:', cap.get(cv2.CAP_PROP_FRAME_WIDTH), ', FrameHeight:',
                              cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        hasFrame, frame = cap.read()
                        while hasFrame and not stopCapturingVideo and not websocket5004.closed:
                            img_encode = cv2.imencode('.jpg', frame)[1]
                            img_encode_byte = img_encode.tobytes()
                            #zip_data = gzip.compress(img_encode_byte) #压缩传
                            await websocket5004.send(img_encode_byte)
                            if isinstance(CameraID,str):# 如果是读文件传的，就慢点儿传
                                await asyncio.sleep(0.167)
                            sentPackets += 1
                            # print('WebSocketServer发送数据包数量：',sentPackets)
                            hasFrame, frame = cap.read()
                        cap.release()
                        if stopCapturingVideo or websocket5004.closed:
                            break
                else:
                    print('请先连接WebSocket5004')
            elif cmd_text['cmd'] == 'stopCapturingVideo':
                print('正在中止视频发送程序...')
                stopCapturingVideo = True
                await websocket5004.send('close')
                await websocket5004.close()
    except Exception as e:
        if websocket5002.closed:
            print('WebSocketServer', port, '端口来自', websocket5002.remote_address, '的连接已关闭')
        else:
            print('WebSocketServer', port, '端口消息处理程序出错已退出，错误原因：', str(e))


# 服务器端主逻辑，未使用
# websocket和path是该函数被回调时自动传过来的，不需要自己传
async def server_main_logic(websocket, path):
    # await check_permit(websocket)
    await recv_msg(websocket)


# 收指令WebSocketServer，服务器端启动时由此启动5002端口
async def server_cmd_port(websocket, path, port=None):
    global ws5002
    print('WebSocketServer', port, '端口收到来自', websocket.remote_address, '的连接请求')
    ws5002=websocket
    await recv_send(ws5002,ws5004,port)

# 发数据WebSocketServer，服务器端启动时由此启动5004端口
async def server_data_port(websocket, path, port=None):
    global ws5004
    print('WebSocketServer',port,'端口收到来自',websocket.remote_address,'的连接请求')
    ws5004 = websocket
    await recv_msg(websocket, port)# 这个函数不起什么实际功能性作用，只是保持WebSocket连接


# 把ip换成自己本地的ip
# start_server = websockets.serve(server_main_logic, '127.0.0.1', 8888)
# 如果要给被回调的main_logic传递自定义参数，可使用以下形式
# 一、修改回调形式
# import functools
# start_server = websockets.serve(functools.partial(server_main_logic, other_param="test_value"), '10.10.6.91', 5678)
# 修改被回调函数定义，增加相应参数
# async def main_logic(websocket, path, other_param)

# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()
# 服务器端代码结束================================================


# 客户端代码================================================
# 向服务器端认证，用户名密码通过才能退出循环，未使用
async def auth_system(websocket):
    while True:
        cred_text = input("please enter your username and password: ")
        await websocket.send(cred_text)
        response_str = await websocket.recv()
        if "congratulation" in response_str:
            return True

# 向服务器端发送认证后的消息，未使用
async def send_msg(websocket):
    while True:
        _text = input("please enter your context: ")
        if _text == "exit":
            print(f'you have enter "exit", goodbye')
            await websocket.close(reason="user exit")
            return False
        await websocket.send(_text)
        recv_text = await websocket.recv()
        print(f"{recv_text}")

# 向服务器端发送指令，5002端口用
async def send_cmd(websocket,command=''):
    await websocket.send(command)


# 等服务器端消息，5004端口用
async def receive_data(videoWebsocket, sendGestureWebsocket=None,svm_hand_model=None, dnn_hand_model=None,
                       myFace=None, handAreaScope=[3000,102400],useWaterShed=True, moveSeg=True,
                       useBackSeg=True,saveRawVideo=None,saveVideo=None):
    global gGESTURE
    global gConfidence
    lastframe = None
    lastGesture = ''  # 记录上一个手势
    sendGesture2ServerViaWebSocket = False  # 是否通过WebSocket向服务器传送手势识别结果的标识
    if GestureSendMode == 'WEBSOCKET' and sendGestureWebsocket is not None:
        sendGesture2ServerViaWebSocket = True

    # 保存视频处理结果用
    vidRaw_writer = None
    vid_writer=None
    Width, Height=640,480
    if saveRawVideo:
        print('创建原始视频存储对象...',end='')
        saveRawVideoFileName=makeVideoFileName(filename=saveRawVideo)
        vidRaw_writer = cv2.VideoWriter(saveRawVideoFileName,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (Width,Height))
        print('OK')
    if saveVideo:
        print('创建处理过的视频存储对象...',end='')
        saveRecVideoFileName = makeVideoFileName(filename=saveVideo)
        vid_writer = cv2.VideoWriter(saveRecVideoFileName,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (Width*2,Height))
        print('OK')

    # 测试用----------------------------------------------------------
    testFPS = False # 测试标记，测试机器人传来图像速率时，设置为True，正常运行程序功能时设置为False
    testSeconds=3 # 测试时长（秒数）
    recv_frames = 0  # 收到的帧数
    startTime= None # 计时起点，用于测试机器人传来图像速率
    # 测试用----------------------------------------------------------

    recvDadaPackets=0 # 记录收到数据包数量
    completePackets=0 # 记录处理完的数据包数量
    k = 0 # 记录处理过的帧数
    print('WebSocket视频数据客户端等待从服务器接收数据...')
    while not videoWebsocket.closed:
        recv_data = await videoWebsocket.recv()
        recvDadaPackets +=1
        k = (k + 1) % 2592000 # 一天重置一次
        # print('WebSocket视频数据客户端收到数据包个数：',recvDadaPackets)
        if testFPS: # 测试帧率
            if startTime is None:
                startTime=time.time() # 第一次收到数据开始计时
                print('测试时间起点：', startTime)
            recv_frames +=1
            currentTime=time.time()
            if currentTime-startTime>0:
                fps='，平均FPS：' + str(round(recv_frames / (currentTime - startTime), 1))
            else:
                fps=''
            print('当前时间：',currentTime,'收到帧数：',recv_frames,fps)
            if currentTime-startTime>=testSeconds:# 统计到设定秒数就结束，自动转为正常状态
                fps_recv=round(recv_frames/(currentTime-startTime),1)
                print(testSeconds,'秒内共收到',recv_frames,'帧数据，平均FPS：',fps_recv,'测试结束，转入正常工作状态。')
                testFPS=False
                # recvDadaPackets =0
        else:
            if recv_data=='close':#仅在自己搭的测试环境有用
                print('WebSocket视频数据客户端收到服务器关闭连接指令，中止接收数据')
                break
            else:
                # print('处理收到的第',recvDadaPackets,'帧数据...')
                #decom_recv_data = gzip.decompress(recv_data) # 解压缩
                jpg_frame = np.frombuffer(recv_data, dtype = 'uint8')
                img = cv2.imdecode(jpg_frame, cv2.IMREAD_COLOR)
                img = cv2.flip(img, 1)

                if saveRawVideo: # 保存原始视频
                    vidRaw_writer.write(img)

                if lastframe is None:
                    lastframe = np.copy(img)
                tc = time.time()
                if k % ProcessOneEverynFrames == 0:
                    gesture, cg, lastframe = frameProcess(img, lastframe,handsvm = svm_hand_model,
                                                          handdnn=dnn_hand_model,
                                                          myFace=myFace,
                                                          handAreaScope = handAreaScope,
                                                          useWaterShed = useWaterShed,
                                                          moveSeg = moveSeg,
                                                          useBackSeg = useBackSeg,
                                                          showVideo = True,
                                                          saveVideo = saveVideo,
                                                          collectHandData = None)
                    completePackets += 1
                    tc=time.time() - tc
                    fps_recg = round(completePackets / tc, 0) # 每秒处理多少帧
                # print('WebSocket数据处理客户端处理完：',completePackets,'帧，本次耗时',round(tc,0),'秒，平均处理速度：',fps_recg,'帧/秒')

                # 如果作为REST服务器对外提供服务，则将识别结果写入全局变量
                    if AsGestureRestServer:
                        lock.acquire()
                        web.gGESTURE = gesture
                        web.gConfidence = cg
                        lock.release()

                    if gesture != lastGesture and gesture != 'invalid':
                # 通过Websocket发送手势识别结果
                        if sendGesture2ServerViaWebSocket:
                            try:
                                gestureJSON = gesture2JSON(gesture, cg)
                                await sendGestureWebsocket.send(gestureJSON)
                            except Exception as e:
                                sendGesture2ServerViaWebSocket = False
                                print("连接异常，手势识别结果将不再通过WebSocket接口发送！")
                    lastGesture = gesture

                    key = cv2.waitKey(1)&0xFF
                    if key == 27:
                        if saveRawVideo:  # 保存原始视频
                            print('正在保存原始视频...', end='')
                            vidRaw_writer.release()
                            print('OK')
                        if saveVideo:  # 保存处理过的视频
                            print('正在保存处理过的视频...', end='')
                            vid_writer.release()
                            print('OK')


# 客户端连接服务器并发送指令
async def client_connect_send(ip,port,cmd=None):
    url='ws://'+ ip + ':' + port
    lastcmd=None
    try:
        async with websockets.connect(url) as websocket:
            while not websocket.closed:
                if cmd is None:
                    while True:
                        if lastcmd is None:
                            text = input("请输入指令序号（1：startCapturingVideo；2：stopCapturingVideo；3：exit）：")
                        else:
                            text = input("按回车键退出...")
                        if lastcmd is None and text=='1':
                            cmd='startCapturingVideo'
                            break
                        elif lastcmd is None and text=='2':
                            cmd = 'stopCapturingVideo'
                            break
                        elif text in ['3',''] :
                            await websocket.close()
                            print('WebSocket视频指令客户端已向服务器发关闭'+str(port)+'端口连接请求')
                            return
                        else:
                            print('输入的指令序号无效！')
                cmd_JSON=cmdJSON(cmd=cmd)
                print('向WebSocket', url, '发送指令', cmd_JSON)
                await send_cmd(websocket,cmd_JSON)
                print('向WebSocket', url, '发送指令完毕')
                lastcmd=cmd
                cmd=None
    except Exception as e:
        print('向',url,'发起WebSocket连接请求出错：',str(e))

# 客户端连接服务器并等待接收数据
async def client_connect_recv(ip,port,svm_hand_model_path=None, dnn_hand_model_path=None,face_cascade_path=None,
                              resFaceModel_path=None, useDlibFace=True,handAreaScope=[3000,102400],useWaterShed=True,
                              moveSeg=True,useBackSeg=True,saveRawVideo=None,saveVideo=None):
    svm_hand_model = None
    if not svm_hand_model_path is None:
        print('加载SVM手形识别模型...', end='')
        if os.path.exists(svm_hand_model_path):
            hand_model = joblib.load(svm_hand_model_path)  # 加载训练好的手形识别svm模型
            print('OK')
        else:
            print('找不到SVM手形识别模型文件')

    dnn_hand_model = None
    if not dnn_hand_model_path is None: # 启用深度学习
        print('加载深度手形识别模型...', end='')
        if os.path.exists(dnn_hand_model_path):
            dnn_hand_model = handDNN(dnn_hand_model_path)
            print('OK')
        else:
            print('深度手形识别模型文件不存在')
    else:
        print('未设定深度手形识别模型文件路径')

    # 加载人脸检测模型
    myFace = faceDection(useDlibFace=useDlibFace, face_cascade_path=face_cascade_path,
                             resFaceModel_path=resFaceModel_path)

    if AsGestureRestServer:
        print('正在启动REST端口服务...')
        tRestServer = Thread(target = RestAPP.run,args=(GestureRestServerPort,), daemon = True)
        tRestServer.start()
        print('REST端口服务启动成功')

    videoURL='ws://' + ip + ':' + port # 获取视频的websocket地址
    keepConnection=True # 保持连接标记，该值为True，则出错时一直试图重新连接
    while keepConnection:
        # try:
            if GestureSendMode=='WEBSOCKET':
                # 发送识别结果的websocket地址
                sendGestureURL = 'ws://' + SendGestureToWebSocketServerIP + ':' + str(SendGestureToWebSocketServerPort)
                print('准备连接',sendGestureURL,'...',end='')
                async with websockets.connect(sendGestureURL, ping_timeout=60) as sendGestureWebsocket:
                    print('连接成功！')
                    print('准备连接', videoURL, '...',end='')
                    async with websockets.connect(videoURL, ping_timeout=60) as videoWebsocket:
                        print('连接成功！')
                        print('连接成功！准备从', videoURL, '接收数据')
                        await receive_data(videoWebsocket=videoWebsocket, sendGestureWebsocket=sendGestureWebsocket,
                                           svm_hand_model=svm_hand_model,dnn_hand_model=dnn_hand_model,
                                           myFace=myFace, handAreaScope=handAreaScope, useWaterShed=useWaterShed,
                                           moveSeg=moveSeg,useBackSeg=useBackSeg,
                                           saveRawVideo=saveRawVideo,saveVideo=saveVideo)
                    keepConnection = False
            else:
                print('准备连接', videoURL, '...', end='')
                async with websockets.connect(videoURL,ping_timeout=60) as videoWebsocket:
                    print('连接成功！')
                    print('准备从', videoURL, '接收数据')
                    await receive_data(videoWebsocket, svm_hand_model=svm_hand_model, dnn_hand_model=dnn_hand_model,
                                       myFace=myFace, handAreaScope=handAreaScope, useWaterShed=useWaterShed,
                                       moveSeg=moveSeg, useBackSeg=useBackSeg, saveRawVideo=saveRawVideo,
                                       saveVideo=saveVideo)
                keepConnection=False
        # except Exception as e:
        #     print('Websocket在通信或数据处理时发生错误：',str(e),' 尝试重新连接......')

# 客户端代码结束================================================

# 用于模拟下发机器人视频流的WebSocket服务器
def StartSimulationWebSocketServers():
    # 默认5002端口用于收指令
    # 默认5004端口用于发视频流
    print('启动Websock服务器'+str(VideoCMPPort)+'端口...')
    start_server5004 = websockets.serve(functools.partial(server_data_port,port=str(VideoCMPPort)), '127.0.0.1', VideoDataPort)
    asyncio.get_event_loop().run_until_complete(start_server5004)
    print('启动Websock服务器'+str(VideoCMPPort)+'端口...')
    start_server5002 = websockets.serve(functools.partial(server_cmd_port,port=str(VideoCMPPort)), '127.0.0.1', VideoCMPPort)
    asyncio.get_event_loop().run_until_complete(start_server5002)
    print('WebSocket服务器启动成功')
    asyncio.get_event_loop().run_forever()
    print('服务器启动程序已退出')

if __name__ == '__main__':
    StartSimulationWebSocketServers()