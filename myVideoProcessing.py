from myGlobalVariables import *
from myCommonModules import text_save,makeVideoFileName
from sklearn.externals import joblib
from myFrameProcessing import frameProcess
from myMapDemo import baiduMap
import socket
import requests

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

def localVideoProcces(cameraID=0, nFrames=2, hand_svm_model=r"..\model\hand_svm_model.m",
                      face_cascade_path=r"..\model\haarcascade_frontalface_alt2.xml",
                      handSize=100,useWaterShed=True, moveSeg=True,
                      GestureSendMode = 'SOCKET',ip=None,port=8888,restURL=None,showVideo=True, saveVideo=None,
                      collectHandData=None,demoMap=True,mapPath=None, chromeDriverPath=None,sensitivity=1):

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

    handsvm = None
    if not hand_svm_model is None:
        if os.path.exists(hand_svm_model):
            handsvm = joblib.load(hand_svm_model)  # 加载训练好的手形识别svm模型

    if demoMap:
        mapDemo = baiduMap(mapPath=mapPath, mapType='BMAP_NORMAL_MAP', chromeDriverPath=chromeDriverPath,sensitivity=sensitivity)
        demoMap = mapDemo.canDemo


    # 读入视频，提取帧图像
    cap = cv2.VideoCapture(cameraID)

    # 强制视频格式转换，防止YUK格式帧率过低
    fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    print('FrameWidth:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),', FrameHeight:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hasFrame, frame = cap.read()
    lastFrame = np.copy(frame)  # 用np.copy(frame)比frame.copy()速度快

    # 加载Haar人脸检测器
    face_cascade = None
    if not face_cascade_path is None:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)  # 加载级联分类器模型
        face_cascade.load(face_cascade_path)

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
    while cap.isOpened():
        k = (k + 1) % 2592000  # 一天重置一次
        t = time.time()
        hasFrame, currentFrame = cap.read()
        if currentFrame is None:
            cv2.waitKey()
            continue

        currentFrame = cv2.flip(currentFrame, 1)

        # 跳帧处理，每nFrames帧做一次处理
        if k % nFrames == 0:
            gesture, cg, lastFrame = frameProcess(currentFrame, lastFrame, handsvm, face_cascade, handSize,
                                                  useWaterShed, moveSeg, showVideo,
                                                  saveVideo=vid_writer, collectHandData=collectHandData)
            if demoMap:
                try:
                    mapDemo.run(command=gesture)  # 操控地图
                    if mapDemo.confirmClose:
                        TellServerClose(lastkey=27, key=27,
                                        sendGesture2ServerViaSocket=sendGesture2ServerViaSocket,
                                        sendGesture2ServerViaREST=sendGesture2ServerViaREST,
                                        socket=c, baseurl=baseurl)
                        break
                except Exception as e:
                    print('执行动作出错：', e)
            if sendGesture2ServerViaSocket:
                try:
                    c.send(gesture.encode("utf8"))
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
                res = requests.get(baseurl, params=params)
                # res.encoding = 'utf-8'
                # print(res.text)

            # 响应键盘，等1ms，按Esc键退出
            key = cv2.waitKey(1)
            if lastkey == 27:
                TellServerClose(lastkey=lastkey,key=key,
                                sendGesture2ServerViaSocket=sendGesture2ServerViaSocket,
                                sendGesture2ServerViaREST=sendGesture2ServerViaREST,
                                socket=c,baseurl=baseurl)
                # 保存手形数据，采集训练数据时用
                if collectHandData:
                    text_save(handdata, collectHandData + '_data.txt')
                break
            lastkey = key
    cap.release()
    if saveVideo:
        vid_writer.release()
        print('手势识别结果视频已经记录在文件', videofilename,'中。')

# 告知服务器结束通信
# lastkey=27,key=-1,按了一次ESC键，发送'end'，告知服务器，客户端终止；
# lastkey=27,key=27,连续按了两次ESC键，发送'exit'，命令服务器程序终止。
# sendGesture2ServerViaSocket=True,通知Socket服务器，socket是Socket对象
# sendGesture2ServerViaREST=False,同时REST服务器，baseurl是服务器网址
def TellServerClose(lastkey=27,key=-1,sendGesture2ServerViaSocket=False,sendGesture2ServerViaREST=False,socket=None,baseurl=None):
    if lastkey == 27:
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