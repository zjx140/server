import cv2
import numpy as np
import socket
from myConfig import *
from myMapDemo import baiduMap


# Socket客户端，连接成功后逐帧向服务器发送视频图像
# cameraID，摄像头ID
# ip, port，服务器IP和端口号
# 每nFrames帧处理一次，以减轻通信压力
# showVideo=True,显示客户端视频
# DemoViaSocket=True，启动客户端手势操纵Demo
# sensitivity手势响应灵敏度
def socketClient(cameraID=0,ip='127.0.0.1',port=6666,nFrames = 2,showVideo=True,DemoViaSocket=True,sensitivity=1):
    if DemoViaSocket:
        mapDemo = baiduMap(mapPath=MapPath,mapType='BMAP_NORMAL_MAP',chromeDriverPath=ChromeDriverPath,sensitivity=sensitivity)
        DemoViaSocket = mapDemo.canDemo

    print('Socket客户端启动，正在连接服务器......')
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn=c.connect_ex((ip,port))
    if conn==10061:
        print('Socket客户端无法与服务器{}连接，已退出。'.format((ip,port)))
        return -1
    print('Socket客户端已与服务器{}连接'.format((ip,port)))

    k = 0 # 记录处理过的帧数
    cap = cv2.VideoCapture(cameraID)
    # 强制视频格式转换，防止YUK格式帧率过低
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    frameWidth=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('FrameWidth:', frameWidth, ', FrameHeight:', frameHeight)

    recieveBUFSIZE = 1024
    sendBUFSIZE = 921600  # 640*480*3
    # sendBUFSIZE = 6220800 # 1920*1080*3
    #sendBUFSIZE = frameWidth * frameHeight * 3

    while (cap.isOpened()):
        hasFrame, frame = cap.read()
        if hasFrame:
            k = (k + 1) % 2592000  # 一天重置一次
            # print(frame.shape)
            if showVideo:
                showframe=cv2.flip(frame,1)
                cv2.imshow('Client',showframe)
            # 跳帧处理，每nFrames帧做一次处理
            if k % nFrames == 0:
                # frame=np.array(frame).tostring() # 不编码，直接转字符串发送

                img_encode = cv2.imencode('.jpg', frame)[1]
                rows, _ = img_encode.shape
                data = np.array(img_encode)
                len0, _ = data.shape
                # 加0补位，不加0会产#生TCP粘包问题,其中BUFFER_SIZE是TCP两端约定的数值，必须相同
                data0 = np.zeros((sendBUFSIZE - len0,1), dtype = np.uint8)
                finaldata = np.vstack((data, data0))  # 合并
                stringdata = finaldata.tostring()  # 变成字符串
                try:
                    c.send(stringdata)
                    recvData = c.recv(recieveBUFSIZE)
                    recvData=recvData.decode('utf-8')
                    # print('客户端向服务器发送了视频数据，收到服务器返回的手势识别结果：',recvData)
                except Exception as e:
                    print("Socket连接异常，可能服务器端已经关闭，客户端已被迫关闭。")
                    break
                if DemoViaSocket:
                    try:
                        mapDemo.run(command=recvData) # 操控地图
                        if mapDemo.confirmClose:
                            break
                    except  Exception as e:
                        print('执行动作出错：',e)

        # 响应键盘，等1ms，按Esc键退出
        key = cv2.waitKey(1)
        if key == 27:break
    c.close()
    return 0

if __name__ == '__main__': # 程序从这儿开始执行
    # socketClient()，启动Socket客户端，采集图像并传给视频分析服务器
    # 参数说明：
    # ip, port，服务器IP和端口号
    # Socket客户端，连接成功后逐帧向服务器发送视频图像
    # 每nFrames帧处理一次，以减轻通信压力
    # showVideo=True,客户端显示视频
    # DemoViaSocket=True，启动客户端手势操纵Demo
    socketClient(cameraID=CameraID, ip=VideoServerIP, port=VideoServerPort, nFrames=ProcessOneEverynFrames,
                 showVideo=True,DemoViaSocket=DemoMap,sensitivity=Sensitivity)
