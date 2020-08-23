# !--*-- coding: utf-8 --*--
import os

####################################
# 通用参数
#===================================
# 摄像头ID
CameraID = 0
#CameraID = r'./video/rawvideo8.avi'
#CameraID =r'../video/real5.avi'
# CameraID =r'../testdata/test1597556718.57631_movedown.avi'
# CameraID =r'C:\Users\Haibo\Desktop\手势数据\王海枫\背景复杂_室内光线不好时.avi' # 肤色范围偏移特别厉害，Cb阈值需要上调
# CameraID =r'C:\Users\Haibo\Desktop\手势数据\刘奇奇\自然光，简单背景，重录.avi'
# CameraID =r'C:\Users\Haibo\Desktop\手势数据\王祉元\白天自然光线+简单背景.avi'
#-----------------------------------
# 深度学习手势识别模型路径，为None时不启用深度学习手形识别
HandShapeDNNPath = r"./model/" # OpenPose的CaffeModel和prototxt路径；MobileNetV2CCDLite模型文件及标签路径
# HandShapeDNNPath = r"../model/hand_pose_model.pth"# OpenPose的Pytorch模型
# HandShapeDNNPath =None

#-----------------------------------
# 手形识别SVM模型路径
# HandShapeSVMPath=r"../model/hand_svm_model.m"
HandShapeSVMPath=None
#-----------------------------------
# 人脸检测模型路径，如果UseDlibFace = True，FaceCascadePath和ResFaceModelPath均忽略
# HARR模型，快，准确度低
# FaceCascadePath = r"../model/haarcascade_frontalface_alt2.xml"
FaceCascadePath = None

# Resnet模型所在路径，准确度最高，较慢
ResFaceModelPath=r"./model/"
# ResFaceModelPath=None

UseDlibFace = False # 使用Dlib HoG人脸检测，速度快，准确度较高
#-----------------------------------
# 手部周长阈值，周长（像素数）在此区间才当候选手处理
HandSize = [200,3000]
# 手部面积阈值
HandAreaScope=[3000,102400] # 3000>2500=(200/4)^2, 102400=640*480/3

# 是否启用背景分割算法，背景与肤色接近时要启用
UseBackSeg=False

# 是否启用运动分割算法，背景与肤色接近时要启用
UseMoveSeg=False

# 是否启用分水岭算，光线差，手部分割不完整时建议启用
UseWaterShed=False

# 启动时是否打开YCrCb调节工具
OpenYCrCbTool=False

# YCrCb分割阈值手动设置
YMinThreshold, YMaxThreshold, YCrMinThreshold, YCrMaxThreshold, YCbMinThreshold, YCbMaxThreshold = 38, 255, 133, 255, 103, 133
# YMinThreshold, YMaxThreshold, YCrMinThreshold, YCrMaxThreshold, YCbMinThreshold, YCbMaxThreshold = None,None,None,None,None,None
#-----------------------------------
# 每ProcessOneEverynFrames帧图像处理一次，以减轻通信和计算压力
ProcessOneEverynFrames = 2
#-----------------------------------
# 视频窗口显示比例
VideoWinScale=1
# 是否显示在顶层
ShowOnTop=False
#-----------------------------------
# 原始视频数据记录路径，设置为None不记录
RawVideoPath=None
#RawVideoPath=r'../video/rawvideo'

# 识别结果视频数据记录路径，设置为None不记录
RecVideoPath=None
# RecVideoPath=r'../video/recvideo'

# 设置测试数据记录路径，不用时设置为None
# RecTestData='../testdata/test'
RecTestData=None
#-----------------------------------
# 是否启动交互地图演示
DemoMap = False

# 人机交互演示地图路径
MapPath = 'file:///'+ os.path.dirname(os.path.dirname(__file__)) + '/mapDemo/demo.html'

# webdriver.Chrome浏览器路径
ChromeDriverPath=r'D:\Program Files\Chrome\chromedriver.exe'

# 手势响应灵敏度，越大越迟钝，每sensitivity次响应一次
Sensitivity= 10
####################################


####################################
# 手势识别由服务器端(gesture_server)处理时
#===================================
# 接收视频的服务器IP地址，客户端（gesture_client）用
VideoServerIP = '127.0.0.1'
# 接收视频的服务器端口，客户端（gesture_client）用
VideoServerPort= 6666
#-----------------------------------
# 服务器端(gesture_server)根据FrameWidth和FrameHeight重构收到的视频帧
# 视频帧宽度像素数,服务器端(gesture_server)用
FrameWidth = 640
# 视频帧高度像素数,服务器端(gesture_server)用
FrameHeight = 480
####################################


####################################
# 手势识别由客户端（gesture_standalone）或MyWebsocket处理时
#===================================
# gesture_standalone或MyWebsocket作为REST服务器提供手势识别结果
AsGestureRestServer = True
# 手势识别服务REST接口开的端口
# gesture_standalone作为REST服务器提供手势识别结果时用
# gesture_server和myWebSocket中，作为REST服务器提供手势识别结果时都用这个设置服务端口
GestureRestServerPort = 8088

# 通过socket、Rest、WebSocket上传手势识别结果时，分别设置为'SOCKET'、'REST'、'WEBSOCKET',客户端（gesture_standalone）用
# GestureSendMode = 'SOCKET'
#GestureSendMode = 'REST'
GestureSendMode = 'WEBSOCKET' # MyWebsocket也用
#GestureSendMode = None

# 通过socket上传时，需设置SendGestureToServerIP和SendGestureToServerPort参数,客户端（gesture_standalone）用
# 接收手势识别结果的服务器IP地址
SendGestureToServerIP = '127.0.0.1'
# 接收手势识别结果的服务器端口
SendGestureToServerPort = 8888

# 通过REST接口上传时，使用GET方法上传识别结果的接口，需设置SendGestureToServerURL参数,客户端（gesture_standalone）用
SendGestureToServerURL='http://127.0.0.1:8080/gesture'

# 通过WEBSOCKET接口上传时，需设置WebSocket服务器IP和端口号
SendGestureToWebSocketServerIP = '127.0.0.1'
SendGestureToWebSocketServerPort = 5000
####################################

####################################
# 手势识别由MyWebsocket处理时
#===================================
# 视频指令通道IP和端口
VideoCMDIP='127.0.0.1'
VideoCMPPort=5002

# 视频数据通道IP和端口
VideoDataIP='127.0.0.1'
VideoDataPort=5004

####################################

####################################
# 接收视频的服务器IP地址和端口，客户端（websocketclient）用
WebSocketServerIP = '127.0.0.1'
WebSocketServerPort = 7777
####################################