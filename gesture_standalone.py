# !--*-- coding: utf-8 --*--
from myConfig import *
from myGlobalVariables import *
from myCommonModules import text_save, makeVideoFileName,myRestAPP,myVideoCapture,gesture2JSON
from myFrameProcessing import frameProcess
from myFaceDetection import faceDection
import requests
from sklearn.externals import joblib
from myMapDemo import baiduMap
import numpy as np
from threading import Lock,Thread
import json
import time
import socket
import cv2
import asyncio
import websockets
from myHandDNN import handDNN
import dlib
if AsGestureRestServer: # 作为REST服务器
    import web


if AsGestureRestServer: # 作为REST服务器
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
            print('接收到REST接口手势识别结果请求，返回：')
            ret = lock.acquire(blocking=False)
            if ret:
                gestureJSON = gesture2JSON(web.gGESTURE, web.gConfidence)
                lock.release()
            print(gestureJSON)
            return gestureJSON

# 视频处理程序，读本地摄像头，识别手势后将结果发给服务器
# cameraID=0，本地摄像头ID
# 每隔nFrames帧处理一次
# hand_svm_model="hand_svm_model.m"，SVM模型路径
# face_cascade_path，人脸检测模型路径
# useWaterShed=True,手部分割时，启用分水岭算法
# moveSeg=True,启用运动分割检测手部
# GestureSendMode = 'SOCKET',向服务发送识别结果的接口方式，'SOCKET'或'REST'
# ip='127.0.0.1',port=8888，接收手势识别结果的服务器IP地址和端口号,ip=None或不传参，则不连接服务器
# restURL=None,通过REST接口项服务器发送手势识别结果时的URL
# wsNone，通过WeBSOCKET上传手势识别结果时的WebSocket连接
# showVideo=True显示视频
# 如果保存视频，用saveVideo=传入保存文件路径
# collectHandData=None不采集手形数据，如需采集手形数据，传入手形名称。
# 在当前目录collectHandData + r'_imgs/'中保存collectHandData + '_' + str(t) + '.jpg'手部图像
# 在当前目录下以collectHandData + '_data.txt'保存特征数据
# 所采集数据供SVM训练用
# demoMap=True,是否启动地图演示
# mapPath，地图网页路径
# chromeDriverPath，ChromeDriver路径
# sensitivity,手势响应灵敏度

async def localVideoProcces(cameraID=0, nFrames=2, hand_svm_model=None, dnn_hand_model = None,face_cascade_path=None, resFaceModel_path=None,
                      useDlibFace=True, handAreaScope=[3000,102400],useWaterShed=True, moveSeg=True,useBackSeg=True,
                      GestureSendMode = None,ip=None,port=8888,restURL=None,ws=None,showVideo=True, saveVideo=None,
                      collectHandData=None,demoMap=True,mapPath=None, chromeDriverPath=None,sensitivity=1):

    # # 记录历史帧，生成测试数据用
    rPoint=0 #recordFrames的指针，循环记录
    recordFramesLen = handTrackLen * ProcessOneEverynFrames
    recordFrames=list([None] * recordFramesLen) # 记录历史帧
    testData=[] # 记录测试数据

    c=None # Socket对象初始化为None，以防REST模式下，调用TellSeverClose函数中socket=c参数出错
    baseurl='' # baseur赋初值，以防Socket模式下，调用TellSeverClose函数中baseurl参数出错

    sendGesture2ServerViaSocket = False # 是否通过Socket向服务器传送手势识别结果的标识
    if GestureSendMode=='SOCKET' and ip is not None:
        print('正在连接服务器(',ip,':',port,')......')
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = c.connect_ex((ip, port))
        if conn == 10061:
            print('无法与服务器(',ip,':',port,')建立Socket连接，手势识别结果无法通过Socket接口发送到服务器端。')
        else:
            sendGesture2ServerViaSocket = True
            print('已与服务器(',ip,':',port,')建立Socket连接，手势识别结果将同步发送到服务器端。')

    sendGesture2ServerViaREST = False # 是否通过REST向服务器传送手势识别结果的标识
    if GestureSendMode=='REST' and restURL is not None:
        print('将通过REST接口向',restURL,'发送手势识别结果.....')
        baseurl = restURL + '?'
        sendGesture2ServerViaREST = True

    sendGesture2ServerViaWebSocket = False  # 是否通过WebSocket向服务器传送手势识别结果的标识
    if GestureSendMode == 'WEBSOCKET' and ws is not None:
        sendGesture2ServerViaWebSocket=True

    handsvm=None
    if not hand_svm_model is None:
        if os.path.exists(hand_svm_model):
            handsvm = joblib.load(hand_svm_model)  # 加载训练好的手形识别svm模型
    dnn_model = None
    if not dnn_hand_model is None:
        if os.path.exists(dnn_hand_model):
            dnn_model = handDNN(dnn_hand_model)

    if demoMap:
        mapDemo = baiduMap(mapPath=mapPath, mapType='BMAP_NORMAL_MAP', chromeDriverPath=chromeDriverPath,sensitivity=sensitivity)
        demoMap = mapDemo.canDemo

    # 读入视频，提取帧图像
    cap=myVideoCapture(cameraID=cameraID)

    # 强制视频格式转换，防止YUK格式帧率过低
    fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    print('FrameWidth:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),', FrameHeight:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hasFrame, frame = cap.read()
    lastFrame = np.copy(frame)  # 用np.copy(frame)比frame.copy()速度快

    # 加载人脸检测模型
    myFace=faceDection(useDlibFace=useDlibFace,face_cascade_path=face_cascade_path,resFaceModel_path=resFaceModel_path)

    # 保存视频处理结果用
    vid_writer = None
    if saveVideo:
        videofilename = makeVideoFileName(filename=saveVideo)
        vid_writer = cv2.VideoWriter(videofilename , cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                     (frame.shape[1] * 2, frame.shape[0]))

    # 创建目录，采集手形数据图像用
    if collectHandData:
        if not os.path.exists(collectHandData + '_imgs'):
            os.makedirs(collectHandData + '_imgs')

    if nFrames <= 1: nFrames = 1
    lastkey = -1  # 用于判断是否连续按下两次ESC键
    k = 0  # 记录处理过的帧数
    lastGesture='' # 记录上一个手势
    while cap.isOpened() and hasFrame:
        k = (k + 1) % 2592000  # 一天重置一次
        t = time.time()
        hasFrame, currentFrame = cap.read()
        if k==1:
            fgMask = frameBackGround.apply(currentFrame)
        if currentFrame is None:
            cv2.waitKey()
            continue

        # 记录历史帧，生成测试数据用
        if RecTestData:
            recordFrames[rPoint] = currentFrame.copy()
            rPoint = (rPoint + 1) % len(recordFrames)

        currentFrame = cv2.flip(currentFrame, 1)

        # 跳帧处理，每nFrames帧做一次处理
        if k % nFrames == 0:
            gesture, cg, lastFrame = frameProcess(currentFrame, lastFrame, handsvm=handsvm, handdnn = dnn_model, myFace=myFace,
                                                  handAreaScope=handAreaScope,
                                                  useWaterShed=useWaterShed, moveSeg=moveSeg,
                                                  useBackSeg=useBackSeg,showVideo=showVideo,
                                                  saveVideo=vid_writer, collectHandData=collectHandData)
            # 保存测试视频用
            if RecTestData and gesture!='invalid' and gesture!=lastGesture:
                st=str(time.time())
                videofilename = makeVideoFileName(filename=RecTestData+st+'_'+gesture)
                testData.append(['test'+st+','+gesture])
                vid_test_writer = cv2.VideoWriter(videofilename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                             (frame.shape[1], frame.shape[0]))
                for i in  range(rPoint+1,len(recordFrames)):
                    if recordFrames[i] is not None:
                        vid_test_writer.write(recordFrames[i])
                for i in  range(0,rPoint+1):
                    if recordFrames[i]  is not None:
                        vid_test_writer.write(recordFrames[i])
                vid_test_writer.release()

            if AsGestureRestServer:# 作为REST服务器
                lock.acquire()
                web.gGESTURE = gesture  # 手势名称
                web.gConfidence = cg  # 置信度
                lock.release()

            if demoMap:
                try:
                    mapDemo.run(command=gesture)  # 操控地图
                    if mapDemo.confirmClose:
                        await TellServerClose(lastkey=27, key=27,
                                              sendGesture2ServerViaSocket=sendGesture2ServerViaSocket,
                                              sendGesture2ServerViaREST=sendGesture2ServerViaREST,
                                              sendGesture2ServerViaWebSocket=sendGesture2ServerViaWebSocket,
                                              socket=c, baseurl=baseurl, ws=ws)
                        break
                except  Exception as e:
                    print('执行动作出错：', e)
            if gesture!=lastGesture and gesture!='invalid': # 重复手势不发，无效手势不发
                if sendGesture2ServerViaSocket:
                    try:
                        c.send(gesture.encode("utf-8"))
                    except Exception as e:
                        print("服务器连接异常，可能服务器端已经关闭，正在尝试重新连接服务器......")
                        conn = c.connect_ex((ip, port))
                        print(conn)
                        if conn == 10061:
                            print('无法与服务器(', ip, ':', port, ')建立Socket连接，手势识别结果无法通过Socket接口发送到服务器端。')
                            sendGesture2ServerViaSocket = False
                        else:
                            sendGesture2ServerViaSocket = True
                            print('已与服务器(', ip, ':', port, ')重新建立Socket连接，手势识别结果将继续同步发送到服务器端。')
                if sendGesture2ServerViaREST:
                    params = {
                        'gesture': gesture,
                        'confidence': cg
                    }
                    #res = requests.get(baseurl, params=params)
                    # res.encoding = 'utf-8'
                    # print(res.text)
                if sendGesture2ServerViaWebSocket:
                    try:
                        json_str = gesture2JSON(gesture, cg)
                        await ws.send(json_str)
                    except Exception as e:
                        sendGesture2ServerViaWebSocket = False
                        print("服务器连接异常，可能服务器端已经关闭，手势识别结果将不再通过WebSocket接口发送到服务器端！")
            lastGesture=gesture # 记录上个手势，不重复发

            # 响应键盘，等1ms，按Esc键退出
            key = cv2.waitKey(1) & 0xFF
            if lastkey == 27:
                await TellServerClose(lastkey=lastkey, key=key,
                                      sendGesture2ServerViaSocket=sendGesture2ServerViaSocket,
                                      sendGesture2ServerViaREST=sendGesture2ServerViaREST,
                                      sendGesture2ServerViaWebSocket=sendGesture2ServerViaWebSocket,
                                      socket=c, baseurl=baseurl, ws=ws)
                # 保存手形数据，采集训练数据时用
                if collectHandData:
                    text_save(handdata, collectHandData + '_data.txt')
                # 保存测试数据
                if RecTestData:
                    text_save(testData, RecTestData + '_data.csv')
                break
            lastkey = key
    cap.release()
    if saveVideo:
        vid_writer.release()
        print('手势识别结果视频已经记录在文件', videofilename,'中。')

async def wsClient(wsIP='127.0.0.1',wsPort=5000):
    url='ws://'+ wsIP + ':' + str(wsPort)
    try:
        async with websockets.connect(url) as ws:
            await localVideoProcces(cameraID=CameraID, nFrames=ProcessOneEverynFrames, hand_svm_model=HandShapeSVMPath,
                              face_cascade_path=FaceCascadePath, resFaceModel_path=ResFaceModelPath,
                              useDlibFace=UseDlibFace,handAreaScope=HandAreaScope,
                              useWaterShed=UseWaterShed, moveSeg=UseMoveSeg, useBackSeg=UseBackSeg,
                              GestureSendMode=GestureSendMode,
                              ip=SendGestureToServerIP, port=SendGestureToServerPort, restURL=SendGestureToServerURL,
                              ws=ws,showVideo=True, saveVideo=RecVideoPath, collectHandData=None,
                              demoMap=DemoMap, mapPath=MapPath, chromeDriverPath=ChromeDriverPath,
                              sensitivity=Sensitivity)
    except Exception as e:
        print(str(e))


# 告知服务器结束通信
# lastkey=27,key=-1,按了一次ESC键，发送'end'，告知服务器，客户端终止；
# lastkey=27,key=27,连续按了两次ESC键，发送'exit'，命令服务器程序终止。
# sendGesture2ServerViaSocket=True,通知Socket服务器，socket是Socket对象
# sendGesture2ServerViaREST=False,通知REST服务器，baseurl是服务器网址
# sendGesture2ServerViaWebSocket=False,通知WebSocket服务器，ws是WebSocket服务器连接
async def TellServerClose(lastkey=27,key=-1,sendGesture2ServerViaSocket=False,sendGesture2ServerViaREST=False,
                    sendGesture2ServerViaWebSocket=False,socket=None,baseurl=None,ws=None):
    if lastkey == 27:
        command = ''
        if key ==-1:
            command='CloseClient'# 按一次ESC，通知服务器，客户端退出
        elif key==27:
            command='CloseServer'# 连续按两次ESC，通知服务器端程序退出

        if sendGesture2ServerViaSocket:
            try:
                socket.send(command.encode("utf8"))
            except Exception as e:
                pass
        if sendGesture2ServerViaREST:
            params = {
                'gesture': command,
                'confidence': 1
            }
            try:
                res = requests.get(baseurl, params=params)
            except Exception as e:
                pass
        if sendGesture2ServerViaWebSocket:
            try:
                await ws.send(command.encode("utf8"))
            except Exception as e:
                pass

if __name__ == '__main__':  # 程序从这儿开始执行
    if AsGestureRestServer: # 作为REST服务器
        print('启动REST接口服务...')
        tRestServer = Thread(target=RestAPP.run,args=(GestureRestServerPort,), daemon=True) # daemon=True,线程会随着主线程退出
        tRestServer.start()
    print('启动手势识别程序，读取本地摄像头数据进行手势分析...')
    if GestureSendMode == 'WEBSOCKET':
        asyncio.get_event_loop().run_until_complete(wsClient(wsIP=SendGestureToWebSocketServerIP,
                                                             wsPort=SendGestureToWebSocketServerPort))
    else:
        # localVideoProcces（）：视频处理程序，读本地摄像头
        # 参数说明：
        # cameraID=0，本地摄像头ID
        # 每隔nFrames帧处理一次
        # hand_svm_model="hand_svm_model.m"，SVM模型路径
        # face_cascade_path，HAAR人脸检测模型路径
        # resFaceModel_path, ResNet人脸检测模型路径
        # useWaterShed=True,手部分割时，启用分水岭算法
        # moveSeg=True,启用运动分割检测手部
        # GestureSendMode，向服务发送识别结果的接口方式，'SOCKET'或'REST'
        # ip,port，接收手势识别结果的服务器IP地址和端口号,ip=None或不传参，则不连接服务器
        # restURL，通过REST接口接收手势识别结果的URL,restURL=None，则不连接服务器
        # ws，WebSocket连接
        # showVideo=True显示视频
        # 如果保存视频，用saveVideo=RecVideoPath传入保存文件路径
        # collectHandData=None不采集手形数据，如需采集手形数据，传入手形名称。
        # 在当前目录collectHandData + r'_imgs/'中保存collectHandData + '_' + str(t) + '.jpg'手部图像
        # 在当前目录下以collectHandData + '_data.txt'保存特征数据
        # 所采集数据供SVM训练用
        # demoMap=True,启动地图交互演示
        # mapPath,地图网页路径
        # chromeDriverPath，ChromeDriver路径
        # sensitivity,手势响应灵敏度
        asyncio.get_event_loop().run_until_complete(localVideoProcces(cameraID=CameraID, nFrames=ProcessOneEverynFrames,
                                                                      hand_svm_model=HandShapeSVMPath,
                                                                      dnn_hand_model = HandShapeDNNPath,
                                                                      face_cascade_path=FaceCascadePath,
                                                                      resFaceModel_path=ResFaceModelPath,
                                                                      useDlibFace=UseDlibFace,
                                                                      handAreaScope=HandAreaScope,
                                                                      useWaterShed=UseWaterShed, moveSeg=UseMoveSeg,
                                                                      useBackSeg=UseBackSeg,
                                                                      GestureSendMode=GestureSendMode,
                                                                      ip=SendGestureToServerIP,
                                                                      port=SendGestureToServerPort,
                                                                      restURL=SendGestureToServerURL,
                                                                      showVideo=True, saveVideo=RecVideoPath,
                                                                      collectHandData=None,
                                                                      demoMap=DemoMap, mapPath=MapPath,
                                                                      chromeDriverPath=ChromeDriverPath,
                                                                      sensitivity=Sensitivity))
    print('手势识别程序已退出')