import socket
import web
import json
from threading import Thread,Lock
from sklearn.externals import joblib

from myConfig import *
from myGlobalVariables import *
from myCommonModules import *
from myFrameProcessing import frameProcess # 单帧处理

# 全局变量，用于REST返回,web.前缀是必须的，否则REST接口返回的都是初值
web.gGESTURE = 'invalid'# 手势名称
web.gConfidence = 1.00 # 置信度

# 全局变量互斥锁
lock = Lock()

# RESTful接口定义
urls = ('/gesture', 'RestAPI')
RestAPP = web.application(urls, globals())
class RestAPI:
    def GET(self):
        print('接收到REST接口手势识别结果请求，返回：')
        data={}
        lock.acquire()
        data["gesname"]=web.gGESTURE
        data["confidence"]=web.gConfidence
        lock.release()
        gestureJSON=json.dumps(data, cls = MyEncoder)
        print(gestureJSON)
        return gestureJSON



# frameBuffer为保存视频帧的全局变量，frameStates记录锁定状态，防读写冲突
# frameBuffer中保存2帧图像，socket收到数据后写入未被读锁定的变量中
# socketServer给未加读锁定（rLock）的frameBuffer元素加上写锁定（wLock），写入收到的图像，解除锁定（wOK）
# videoProcess给标记为wOK(写好)的frameBuffer元素上锁（rLock），读取一帧图像，然后解锁(rOK)
frameBuffer = [None,None]
frameStates = ['wLock','wLock']


# 将视频帧写入全局缓冲区，供videoProcess用
def writeFrameBuffer(frame):
    global frameStates,frameBuffer
    writeOK = True
    while writeOK:
        for i in [0,1]:  # 读写都优先考虑frameBuffer[0]，冲突才考虑frameBuffer[1]
            if frameStates[i] != 'rLock':
                frameStates[i] = 'wLock'
                frameBuffer[i] = np.copy(frame)
                frameStates[i] = 'wOK'
                writeOK = False
                break
        cv2.waitKey(1)


# 读取全局缓冲区中的图像
def readFrameBuffer():
    global frameStates,frameBuffer
    frame = None
    for i in [0,1]:  # 读写都优先考虑frameBuffer[0]，冲突才考虑frameBuffer[1]
        if frameStates[i] == 'wOK':
            frameStates[i] = 'rLock'
            frame=np.copy(frameBuffer[i])
            frameStates[i] = 'rOK'
            break
    return frame

# Socket服务器,Socket接收数据写入缓冲区，适用于多线程处理
# ip和port为侦听IP地址和端口号
# BUFSIZE是接收缓冲区大小
# BUFSIZE = 921600 # 640*480*3
# BUFSIZE = 6220800 # 1920*1080*3
# Height=480,Width=640 是视频分辨率
# ShowVideo决定是否开窗口显示收到的视频
def socketServer(ip='127.0.0.1', port=6666,BUFSIZE = None ,Height=480,Width=640,ShowVideo=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 重用IP和端口号
    s.bind((ip,port))
    s.listen(1)
    print('Socket服务器端启动，开始侦听......')
    if BUFSIZE is None:
        BUFSIZE=Height * Width * 3 # Socket接收数据缓冲区大小
    while True:
        conn, addr = s.accept()
        print('接到来自%s的连接' % addr[0])
        while True:
            # 响应键盘，等1ms，按Esc键退出
            key = cv2.waitKey(1)
            if key == 27: break
            try:
                frame = conn.recv(BUFSIZE)
                r=1
                while len(frame)< BUFSIZE and r<BUFSIZE:
                    frame += conn.recv(BUFSIZE-len(frame))
                    r +=1
                if r>=BUFSIZE:# 循环了很多次都没有接收到数据
                    print('长时间接收不到客户端数据，断开连接')
                    s.listen(1)
                    print('Socket服务器端重新开始侦听......')
                    break
                conn.send('OK'.encode('utf-8'))
            except:
                print('客户端断开连接')
                s.listen(1)
                print('Socket服务器端重新开始侦听......')
                break
            if len(frame) == 0: break
            # print('收到视频流数据{}字节'.format(len(frame)))
            frame=np.fromstring(frame, dtype='uint8')
            if len(frame) == Height * Width * 3:
                frame=np.reshape(frame,(Height,Width,3))
                writeFrameBuffer(frame)# 写入全局缓冲区
                if ShowVideo:
                    cv2.imshow('Server',frame)
        conn.close()
        # 响应键盘，等1ms，按Esc键退出
        key = cv2.waitKey(1)
        if key == 27:break
    s.close()
    return 0

# 带视频处理功能的Socket服务器,可做主程序从这里启动
# ip和port为侦听IP地址和端口号，BUFSIZE是接收缓冲区大小
# BUFSIZE = 921600 # 640*480*3
# BUFSIZE = 6220800 # 1920*1080*3
# Height=480,Width=640 是视频分辨率
# ShowVideo决定是否开窗口显示收到的视频
# hand_svm_model="hand_svm_model.m"手形识别SVM模型
# face_cascade_path,人脸检测模型路径
# moveSeg=True 启用运动分割
# saveRawVideo=None不保存原始视频，如需保存，传入保存文件路径'xxx.avi'
# saveVideo=None不保存处理过的视频，如需保存，传入保存文件路径'yyy.avi'
# collectHandData=None不采集手形数据，如需采集手形数据，传入手形名称。所采集数据供SVM训练用
def socketServerWithVideoProcess(ip='127.0.0.1', port=6666,BUFSIZE = None,Height=480,Width=640,showVideo=True,
                                 hand_svm_model=r"..\model\hand_svm_model.m",
                                 face_cascade_path=r"..\model\haarcascade_frontalface_alt2.xml",handSize=100,
                                 useWaterShed=True,moveSeg=True,saveRawVideo=None,saveVideo=None,collectHandData=None):
    global gGESTURE
    global gConfidence

    print('加载SVM手形识别模型...',end='')
    handsvm = None
    if not hand_svm_model is None:
        if os.path.exists(hand_svm_model):
            handsvm = joblib.load(hand_svm_model)  # 加载训练好的手形识别svm模型
    print('OK')

    # 加载Haar人脸检测器
    face_cascade=None
    if not face_cascade_path is None:
        print('加载人脸检测模型...',end='')
        face_cascade = cv2.CascadeClassifier(face_cascade_path)  # 加载级联分类器模型
        face_cascade.load(face_cascade_path)
        print('OK')

    # 保存视频处理结果用
    vidRaw_writer = None
    vid_writer=None
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

    # 创建目录，采集手形数据图像用
    if collectHandData:
        if not os.path.exists(collectHandData+'_imgs'):
            print('创建存储手形图像的文件夹'+collectHandData+'_imgs'+'...', end='')
            os.makedirs(collectHandData+'_imgs')
            print('OK')

    print('启动Socket服务器...',end='')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 重用IP和端口号
    s.bind((ip, port))
    s.listen(1)
    print('OK')
    print('Socket服务器端已启动，开始侦听客户端连接......')
    lastFrame = None
    if BUFSIZE is None:
        BUFSIZE=Height * Width * 3 # Socket接收数据缓冲区大小
    while True:
        conn, addr = s.accept()
        print('接到来自%s的Socket连接' % addr[0])
        while True:
            # 响应键盘，等1ms，按Esc键退出
            key = cv2.waitKey(1)
            if key == 27:
                print('正在关闭Socket连接...',end='')
                conn.close()
                print('OK')
                print('正在关闭Socket服务器...',end='')
                s.close()
                print('OK')
                if saveRawVideo:  # 保存原始视频
                    print('正在保存原始视频...',end='')
                    vidRaw_writer.release()
                    print('OK')
                if saveVideo: # 保存处理过的视频
                    print('正在保存处理过的视频...',end='')
                    vid_writer.release()
                    print('OK')
                # 保存手形数据，采集训练数据时用
                if collectHandData:
                    print('正在保存手形数据...', end='')
                    text_save(handdata, collectHandData + '_data.txt')
                    print('OK')
                return
            try:
                frame = conn.recv(BUFSIZE)
                r=1
                while len(frame)< BUFSIZE and r<BUFSIZE:
                    frame += conn.recv(BUFSIZE-len(frame))
                    r +=1
                if r>=BUFSIZE:# 循环了很多次都没有接收到数据
                    print('长时间接收不到客户端数据，断开连接')
                    s.listen(1)
                    print('Socket服务器端重新开始侦听......')
                    break
            except Exception as e:
                print('客户端断开连接')
                s.listen(1)
                print('Socket服务器端重新开始侦听......')
                break
            if len(frame) > 0:
                # print('收到视频流数据{}字节'.format(len(frame)))
                frame=np.fromstring(frame, dtype='uint8')
                if len(frame) == Height * Width * 3:
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    # frame=np.reshape(frame,(Height,Width,3)) # 如果不用编码解码，需打开此语句
                    frame = cv2.flip(frame, 1)
                    if saveRawVideo: # 保存原始视频
                        vidRaw_writer.write(frame)
                    if lastFrame is None:
                        lastFrame=np.copy(frame)
                    gesture, cg, lastFrame = frameProcess(frame, lastFrame, handsvm=handsvm, face_cascade=face_cascade,
                                                          useWaterShed=useWaterShed,moveSeg=moveSeg,handSize=handSize,
                                                          showVideo=showVideo,saveVideo=vid_writer,
                                                          collectHandData=collectHandData)
                    lock.acquire()
                    web.gGESTURE = gesture # 手势名称
                    web.gConfidence = cg  # 置信度
                    lock.release()
                    try:
                        conn.send(gesture.encode('utf-8'))
                    except:
                        print('客户端断开连接')
                        s.listen(1)
                        print('Socket服务器端重新开始侦听......')
                        break


# 带Socket通信的视频处理程序，从Socket缓冲区获取视频，适用于多线程处理
# 每隔nFrames帧处理一次
# hand_svm_model="hand_svm_model.m"，手形识别SVM模型
# face_cascade_path="haarcascade_frontalface_alt2.xml",人脸检测模型路径
# 如果保存视频，用saveVideo=传入保存文件路径
def socketVideoProcces(nFrames=1, hand_svm_model=r"..\model\hand_svm_model.m",
                       face_cascade_path=r"..\model\haarcascade_frontalface_alt2.xml",saveVideo=None):
    print('Socket视频处理现成已启动')
    handsvm = None
    if not hand_svm_model is None:
        if os.path.exists(hand_svm_model):
            handsvm = joblib.load(hand_svm_model)#加载训练好的手形识别svm模型

    # 读入视频，提取帧图像
    frame = readFrameBuffer()
    while frame is None:
        frame = readFrameBuffer()
        # 响应键盘，等1ms，按Esc键退出
        key = cv2.waitKey(1)
        if key == 27: return

    lastFrame = np.copy(frame)  # 用np.copy(frame)比frame.copy()速度快

    # 加载Haar人脸检测器
    face_cascade = None
    if not face_cascade_path is None:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)  # 加载级联分类器模型
        face_cascade.load(face_cascade_path)

    # 保存视频处理结果用
    vid_writer=None
    if saveVideo:
        vid_writer = cv2.VideoWriter(saveVideo,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1]*2,frame.shape[0]))

    if nFrames <= 1: nFrames = 1
    k = 0 # 记录处理过的帧数
    while True:
        k = (k+1) % 2592000 # 一天重置一次
        t = time.time()
        currentFrame = readFrameBuffer()
        # print('获取视频', '成功' if not currentFrame is None else '失败')

        if currentFrame is None:
            cv2.waitKey(1)
            continue

        currentFrame = cv2.flip(currentFrame,1)

        # 跳帧处理，每nFrames帧做一次处理
        if k % nFrames ==0:
            gesture,cg,lastFrame = frameProcess(currentFrame, lastFrame, handsvm,face_cascade,useWaterShed=True,moveSeg=True, showVideo=True,saveVideo=vid_writer)

             # 响应键盘，等1ms，按Esc键退出
            key = cv2.waitKey(1)
            if key == 27:
                # 保存手形数据，采集训练数据时用
                # text_save(handdata,'data.txt')
                break
    if saveVideo:
        vid_writer.release()


# 监测全局变量赋值情况，调试程序用
def varMonitor():
    while 1:
        print('监测手势识别结果：',gGESTURE)


if __name__ == '__main__': # 程序从这儿开始执行
    print('启动REST接口服务...')
    tRestServer = Thread(target=RestAPP.run,daemon=True) # daemon=True,线程会随着主线程退出
    tRestServer.start()

    # 启动监控全局变量线程，调试用
    # tMonitor=Thread(target=varMonitor,daemon=True)
    # tMonitor.start()

    # Socket接收视频并处理，多线程
    tSocketVideoProcces = Thread(target=socketServerWithVideoProcess(ip='0.0.0.0', port=VideoServerPort,
                                                                     BUFSIZE=None, Height=FrameHeight, Width=FrameWidth,
                                                                     showVideo=True,hand_svm_model=HandShapeSVMPath,
                                                                     handSize=HandSize,
                                                                     face_cascade_path=FaceCascadePath,
                                                                     useWaterShed=True,moveSeg=True, saveRawVideo=None,
                                                                     saveVideo=None,collectHandData=None))
    tSocketVideoProcces.start()

    # Socket接收视频并处理，单线程
    # socketServerWithVideoProcess(ip=VideoServerIP, port=VideoServerPort, BUFSIZE=None, Height=FrameHeight,
    #                              Width=FrameWidth, showVideo=True, hand_svm_model=HandShapeSVMPath,
    #                              handSize=HandSize,
    #                              face_cascade_path=FaceCascadePath,
    #                              useWaterShed=True,moveSeg=True, saveRawVideo=None, saveVideo=None,
    #                              collectHandData=None)


    ## Socket通信与Video处理多线程并行
    # tSocketServer=Thread(target=socketServer)
    # tSocketServer.start()
    #
    # tSocketVideoProcces = Thread(target=socketVideoProcces)
    # tSocketVideoProcces.start()

    # print('关闭REST接口服务...OK')
    print('手势识别程序已退出')
