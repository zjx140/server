# !--*-- coding: utf-8 --*--
import math
import cv2
import os
import numpy as np
import json
import web
import platform
from matplotlib import pyplot as plt
from myGlobalVariables import myTrackBar, myPlot,OpenYCrCbTool,CameraID

# 设置numpy浮点运算错误的报错方式
def setFloatErrWarning():
    np.seterr(all=None, divide=None, over=None, under=None, invalid=None)
    # {‘ignore’, ‘warn’, ‘raise’, ‘call’, ‘print’, ‘log’}
    np.seterr(invalid='ignore')

# 将列表数据写入文本文件，
# filename为写入CSV文件的路径，data为要写入数据列表
def text_save(data, filename='handdata.txt'):
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        # s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        s = s.replace("'", '') + '\n'  # 去除单引号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

# 检查视频文件是否已存在，如果存在，自动增加编号重命名，返回不重名的文件名
# filename，文件路径，不含扩展名
def makeVideoFileName(filename='video'):
    # 如果文件已经存在，自动重命名
    filename = os.path.abspath(filename)
    videofilename = filename + '.avi'
    if os.path.exists(videofilename):
        i = 1
        videofilename = filename + str(i) + '.avi'
        while os.path.exists(videofilename):
            i += 1
            videofilename = filename + str(i) + '.avi'
        print('视频文件已经存在，自动重命名为：', videofilename)
    elif not os.path.exists(os.path.dirname(videofilename)):# 如果目录不存在，创建一个
        print('创建存储视频的文件夹',os.path.dirname(videofilename))
        os.makedirs(os.path.dirname(videofilename))
    return videofilename


# 通过形态学操作平滑二值图像bin_img，返回二值图像
# 先用半径为small_kernel_size的结构元素处理（默认开操作去噪，调大该参数去噪效果好），设置为0时跳过该操作
# 再用半径为big_kernel_size的结构元素处理（默认闭操作填充空隙，调大该参数填充效果好），设置为0时跳过该操作
# smallStruct=cv2.MORPH_ELLIPSE,bigStruct=cv2.MORPH_ELLIPSE,设置结构元素，默认为椭圆
# smallMorph=cv2.MORPH_OPEN,bigMorph=cv2.MORPH_CLOSE,设置形态学操作
# 结构元素可以用small_kernel和big_kernel参数传入（循环中可以避免重复构造，节省时间）
def polish(bin_img, small_kernel_size=5, big_kernel_size=5, smallStruct=cv2.MORPH_ELLIPSE, bigStruct=cv2.MORPH_ELLIPSE,
           smallMorph=cv2.MORPH_OPEN, bigMorph=cv2.MORPH_CLOSE, small_kernel=None, big_kernel=None):
    if small_kernel_size > 0:
        if small_kernel is None:
            small_kernel = cv2.getStructuringElement(smallStruct, (small_kernel_size, small_kernel_size))
        bin_img = cv2.morphologyEx(bin_img, smallMorph, small_kernel)  # 默认开操作去噪
    if big_kernel_size > 0:
        if big_kernel is None:
            big_kernel = cv2.getStructuringElement(bigStruct, (big_kernel_size, big_kernel_size))
        bin_img = cv2.morphologyEx(bin_img, bigMorph, big_kernel)  # 默认闭操作填充
    return bin_img


# 计算输入轮廓contour的中心点坐标(center_x, center_y)
def getCenterXY(contour):
    center_x, center_y = -1, -1  # 找不到中心就返回-1
    M = cv2.moments(contour)  # 计算轮廓的各阶矩,字典形式
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    return (center_x, center_y)


# 直方图均衡化，彩色图像和灰度图像自适应，未用
def AdaptiveEqualizeHist(frame):
    if frame.shape[2] == 3:  # 彩色图像
        (b, g, r) = cv2.split(frame)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # 合并每一个通道
        eFrame = cv2.merge((bH, gH, rH))
    else:  # 灰度图像
        eFrame = cv2.equalizeHist(frame)
    return eFrame


# 按指定大小或缩放比例显示图像
# Img要显示的图像
# WindowName='ShowImage'窗口名称
# ImgSize=(ImgWidth,ImgHeight)指定窗口大小的元组
# xScale=1,yScale=None，或者指定缩放比例
# x=None,y=None，窗口位置
# showOnTop=False，是否显示在顶层
def showImg(Img, WindowName='ShowImage', ImgSize=(0, 0), xScale=1, yScale=None, x=None, y=None, showOnTop=False):
    if yScale is None:
        yScale = xScale
    if xScale!=0:
        cv2.imshow(WindowName, cv2.resize(Img, ImgSize, fx=xScale, fy=yScale))
        if showOnTop: # 显示在顶层
            if platform.system() == 'Windows' and int(cv2.__version__.split('.')[0])>=4:# Windows且penCV 4以上支持
                cv2.setWindowProperty(WindowName,cv2.WND_PROP_TOPMOST,1)
        if x is not None and y is not None:
            cv2.moveWindow(WindowName, x, y)


# Ycrcb空间肤色分割，自动阈值分割，效果好
# Img，输入BGR图像
# YMinThreshold, YMaxThreshold, YCrMinThreshold, YCbMinThreshold, YCrMaxThreshold, YCbMaxThreshold，Y、Cr和Cb通道肤色阈值下限和上限
# faces，人脸区域
# 返回值：
# ycrcbFrame, YCrCb图像
# ycrFrame, Cr通道图像
# ycrcbHand，肤色区域掩膜
def YCrCbSeg(Img, YMinThreshold = 0, YMaxThreshold =255,YCrMinThreshold=125, YCrMaxThreshold=175, YCbMinThreshold=80, YCbMaxThreshold=160,faces=[]):
    ycrcbFrame = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)  # 转换到YCrCb空间
    ycrcbFrame = cv2.GaussianBlur(ycrcbFrame, (5, 5), 0)  # 高斯滤波

    # # 调参用========================================
    # # 绘制颜色直方图，用以确定肤色分割阈值，调参用
    if OpenYCrCbTool:
        # 调节颜色阈值，调参用
        global myTrackBar
        if myTrackBar is None: # 刚打开OpenYCrCbTool开关，尚未创建工具条对象
            trackbar=[('Histogram',0, 1),('Image',0,1),('minY',YMinThreshold,255),('maxY',YMaxThreshold,255),('minCr',YCrMinThreshold,255),('maxCr',YCrMaxThreshold,255),('minCb',YCbMinThreshold,255),('maxCb',YCbMaxThreshold,255)]
            myTrackBar=trackBar(trackbar)

        if myTrackBar.getValue('minCr')!=-1: # 工具窗口关闭后，getValue会返回-1
            YCrMinThreshold = myTrackBar.getValue('minCr')
            YCrMaxThreshold = myTrackBar.getValue('maxCr')
            YCbMinThreshold = myTrackBar.getValue('minCb')
            YCbMaxThreshold = myTrackBar.getValue('maxCb')
            YMinThreshold = myTrackBar.getValue('minY')
            YMaxThreshold = myTrackBar.getValue('maxY')

        if myTrackBar.getValue('Histogram')==1: # 是否打开颜色直方图工具
            global myPlot
            if myPlot is None:
                myPlot=plotHist(ycrcbFrame, None, plt, ('Y', 'Cr', 'Cb'),color=('gray','red','blue'))
            myPlot.histplt.cla()
            myPlot.updateHist(ycrcbFrame)
            myPlot.histplt.pause(0.01)
        else:
            if myPlot is not None:
                myPlot.histplt.close()
                myPlot=None

    # # 调参用========================================

    ycrFrame = np.copy(ycrcbFrame[:, :, 1])  # 转换到YCrCb空间后取Cr通道
    thresCr, ycrcbHand = cv2.threshold(ycrFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print('YCrCb空间自动肤色分割阈值：',thresCr,'YCrMinThreshold',YCrMinThreshold)
    if OpenYCrCbTool:# 手动调节时，按手动设置的得参数分割
        ycrcbHand = cv2.inRange(ycrcbFrame, np.array([YMinThreshold, YCrMinThreshold, YCbMinThreshold]),
                                np.array([YMaxThreshold, YCrMaxThreshold, YCbMaxThreshold]))  # 肤色分割
    else:
        if thresCr < YCrMinThreshold:  # 当自动分割阈值thres低于阈值（常取130）时，会把背景分割进来，此刻采用固定阈值分割
            ycrcbHand = cv2.inRange(ycrcbFrame, np.array([YMinThreshold, YCrMinThreshold, YCbMinThreshold]),
                                    np.array([YMaxThreshold, YCrMaxThreshold, YCbMaxThreshold]))  # 肤色分割
    if OpenYCrCbTool:
        if myTrackBar.getValue('Image')==1:  # 是否显示分割结果图像
            showImg(ycrcbHand,'YCrCb Segmentation',xScale=0.5)
        elif  cv2.getWindowProperty('YCrCb Segmentation',0)!=-1:
            cv2.destroyWindow('YCrCb Segmentation')
    # 把面部肤色区域抹去（置0）
    for (x, y, w, h) in faces:
        margin=0.1 # 向外扩10%的余量
        x1=int(x - margin * w)
        x2=int(x + (1 + margin) * w)
        y1=int(y - (0.1 + margin) * h) # 把额头也抹掉
        y2=int(y + (1.5 + margin) * h) # 把脖子也抹掉
        cv2.rectangle(ycrcbFrame, (x1, y1), (x2, y2), 0, -1)  # 把抹掉人脸
        cv2.rectangle(ycrFrame, (x1, y1), (x2, y2), 0, -1)  # 把抹掉人脸
        cv2.rectangle(ycrcbHand, (x1, y1), (x2, y2), 0, -1)  # 把抹掉人脸
    # cv2.imshow('YCrHand',ycrcbHand)
    return ycrcbFrame, ycrFrame, ycrcbHand


# 分水岭算法分割图像
# Img,彩色图像，本程序中为YCrCb空间彩色图像
# bwImg，二值图像，白色为前景。如果未传入bwImg，则Img要传入BGR图像
# 返回值: markers，分割结果图像，-1为边界轮廓，其他值为区域编号
def WaterShed(Img, bwImg=None):
    # 可能会出现numpy浮点运算invalid错误，打开下面这行语句可以屏蔽错误
    # np.seterr(all='ignore')

    if bwImg is None:
        # YCrCb颜色空间肤色分割，bwImg为肤色区域掩膜
        _, _, bwImg = YCrCbSeg(Img, YCrMinThreshold=130, YCrMaxThreshold=150,
                               YCbMinThreshold=115, YCbMaxThreshold=125)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bwImg, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀，把边界地区都变成白色
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)#距离变换
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)  # 获取前景区域
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)  # 边界区域，此区域和轮廓区域的关系未知
    ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号，序号为 0 — N-1
    markers = markers + 1  # OpenCV 分水岭算法对物体做的标注必须都大于1，背景为标号为0。因此对所有markers加1变成1-N 范围
    markers[unknow == 255] = 0  # 边界区域设置为0
    markers = cv2.watershed(Img, markers)  # 分水岭分割，所有轮廓的像素点被标注为 -1
    return markers

# 自定义轮廓提取函数，可以适应不同的OpenCV版本，只返回轮廓
def myFindContours(img,mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    if cv2.__version__.split('.')[0]=='3':
        _,contours, _ = cv2.findContours(img, mode, method)
    else:
        contours, _ = cv2.findContours(img, mode, method)
    return contours


# 自定义JSON编码器
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# 组装机器人视频请求指令
def cmdJSON(cmd='startCapturingVideo',code=0,external='',message='message'):
        # REQUEST协议格式
        # {
        # “code”：””
        # “cmd”：”startCapturingVideo”
        # “external”:””
        # “message”:””
        # }
        data={}
        data["code"]=code
        data["cmd"]=cmd
        data["external"] = external
        data["message"] = message
        cmd_JSON=json.dumps(data, cls=MyEncoder)
        return cmd_JSON

# 组装手势识别结果响应Json字符串
def gesture2JSON(gesture,confidence=1.0):
        data={}
        data["gesname"]=gesture
        data["confidence"]=confidence
        ges_JSON=json.dumps(data, cls=MyEncoder)
        return ges_JSON

# 计算距离,point格式为(x,y)
# 本函数可以处理点坐标空值（None）的情况
def distance(point1,point2):
    if point1 is None or point2 is None:
        return None
    else:
        return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

# 判断两个数之间的关系
# 本函数可以处理空值（None）情况
def compare(a,b):
    if a is None or b is None:
        return None
    elif a > b:
        return 'larger'
    elif a < b:
        return 'less'
    elif a == b:
        return 'equal'

# 计算3点夹角
def angle(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    if point_1 is None or point_2 is None or point_3 is None:
        return None
    else:
        a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
        b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
        c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
        # A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
        B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
        # C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
        return B

# 计算圆与直线交点
# c和r是圆心和半径
# PointA,PointB是线段端点
def crossCircleLine(c,r,PointA,PointB):
    cx, cy = c[0], c[1]
    ax, ay = PointA[0], PointA[1]
    bx, by = PointB[0], PointB[1]

    m=ax-bx
    if m==0: # 无斜率
        delt =r**2-(ax-cx)**2
        if delt>=0: #两个解
            x1 = ax
            x2 = ax
            s = np.sqrt(r**2-(ax-cx)**2)
            y1 = cy - s
            y2 = cy + s
        else: # 无解
            x1,x2,y1,y2=None,None,None,None
    else: # 有斜率
        k = (by-ay)/(bx-ax)
        b = ay - k * ax
        l = b - cy
        delt = (2 * k * l - 2 * cx) ** 2 - 4 * (k ** 2 + 1)*(cx ** 2 + l ** 2 - r ** 2)

        if delt>=0:#有解
            x1 = (2 * cx - 2 * k * l - np.sqrt(delt))/2/(k**2+1)
            x2 = (2 * cx - 2 * k * l + np.sqrt(delt))/2/(k**2+1)
            y1 = k * x1 + b
            y2 = k * x2 + b
        else:# 无解
            x1, x2, y1, y2 = None, None, None, None
    if not x1 is None:
        p1,p2=(np.int(x1), np.int(y1)), (np.int(x2), np.int(y2))
    else:
        p1,p2=None,None
    return  p1,p2

# 计算两直线交点
# 已知一条直线的两个端点PointA,PointB和另一条直线的斜率k和直线上一点PointC
def crossLine(PointA,PointB,k, PointC):
    ax, ay = PointA[0], PointA[1]
    bx, by = PointB[0], PointB[1]
    cx, cy = PointC[0], PointC[1]

    if ax==bx: # AB无斜率
        x=ax
        y=cy
    else:
        kab=(ay-by)/(ax-bx)
        x=(cy-ay-k*cx+kab*ax)/(kab-k)
        y=cy+k*(x-cx)
    crossPoint=(x,y)
    return crossPoint


# 调节工具条
class trackBar:
    # 初始化
    # 输入：
    # trackBars，工具条设置元组，(barName,initV,maxV,callBack),callBack可省略
    # windowName='TrackBar' 工具条窗口名称
    def __init__(self,trackBars, windowName='YCrCb Adjustment'):
        self.windowName=windowName
        cv2.namedWindow(self.windowName,flags=cv2.WINDOW_NORMAL)
        for bar in trackBars:
            if len(bar)==4:
                cv2.createTrackbar(bar[0], windowName, bar[1], bar[2], bar[3])
            elif len(bar)==3: #未传入回调函数callBack这个参数
                cv2.createTrackbar(bar[0], windowName, bar[1], bar[2], self.callback)

    # 回调函数，什么也不干，只是个摆设
    def callback(self,pos):
        pass

    # 获取trackbar的值，barName是trackbar的名字
    def getValue(self,barName):
        return cv2.getTrackbarPos(barName, self.windowName)

# 绘制颜色直方图
class plotHist():
    # img: 图像，可以是彩色图像或灰度图像
    # cvt：颜色空间转换参数，用cv2.COLOR_XXX2YYY传入，可以取None
    # histplt：plot句柄
    # labels：曲线标签
    # title='Histogram'：直方图标题
    # subplot_rows=1,subplot_cols=1,subplot=1,subtitle=''：子图行数、列数、本子图编号、标题
    # plotlegend=True：是否绘制图例，多个直方图绘制到一个子图时，自第2个起调用时应设置plotlegend=Flase
    # color=('blue', 'green', 'red')：曲线颜色
    # xlim=[0,255]：横坐标刻度范围
    # plotall=True,plot=0：是否显示全部曲线，如果不是，按plot指定的序号显示
    def __init__(self,img,cvt,histplt, labels, title='Histogram',
                  subplot_rows=1, subplot_cols=1,subplot=1,subtitle='',
                  plotlegend=True, color=('blue', 'green', 'red'),
                  xlim=[0,255], plotall=True, plot=0):
        self.cvt = cvt
        self.histplt=histplt
        self.labels=labels
        self.title=title
        self.subplot_rows=subplot_rows
        self.subplot_cols=subplot_cols
        self.plotlegend=plotlegend
        self.color=color
        self.xlim=xlim
        self.plotall=plotall
        self.plot=plot

        self.histplt.figure(title)
        # 先将传入图像转换到指定的颜色空间
        if cvt is None:
            colorspace = img
        else:
            colorspace = cv2.cvtColor(img, cvt)

        self.histplt.subplot(subplot_rows, subplot_cols, subplot)
        self.histplt.xlim(xlim)
        self.histplt.title(subtitle)
        colors=len(color)
        channels=colorspace.shape
        lc=len(channels)# lc==2是灰度图，lc==3是彩色图
        # 判断颜色数与通道数是否一致:如果(图是灰度图但但颜色数不是1)or(图是彩色图但颜色数与图像通道数不一致)
        if (lc==2 and colors !=1) or (lc!=2 and colors != channels[2]):
            print('The number of colors is not equal to the number of channels. The '+ title+ ' cannot be plotted.')
            return
        for i, c in enumerate(color):
            if plotall or i==plot:
                hist = cv2.calcHist([colorspace], [i], None, [256], [0, 255])
                self.histplt.plot(hist, color=c, label=labels[i])
                if plotlegend:
                    self.histplt.legend()
    # 更新直方图曲线
    def updateHist(self,img,subplot=1,subtitle=''):
        if self.subplot_rows!=1 and self.subplot_cols!=1:
            self.histplt.subplot(self.subplot_rows, self.subplot_cols, subplot)
            self.histplt.title(subtitle)
        # 先将传入图像转换到指定的颜色空间
        if self.cvt is None:
            colorspace = img
        else:
            colorspace = cv2.cvtColor(img, self.cvt)
        self.histplt.xlim(self.xlim)
        for i, c in enumerate(self.color):
            if self.plotall or i==self.plot:
                hist = cv2.calcHist([colorspace], [i], None, [256], [0, 255])
                self.histplt.plot(hist, color=c, label=self.labels[i])
                if self.plotlegend:
                    self.histplt.legend()


# 判断一个点point是否落在多个矩形rect内
# 输入：
# 矩形区域列表rect=[[x, y, w, h],[x, y, w, h],...]，与OpenCV人脸检测返回结果同构
# 点point=(x,y)
# margin = 0，边界余量，正为向外扩margin倍，负为向内缩margin倍
# 输出inrect=True 或False
def inRect(rect,point,margin = 0):
    inrect = False  # 标识轮廓中心是否在J矩形内部
    for (x, y, w, h) in rect:
        if x - margin * w < point[0] < x + (1 + margin) * w \
                and y - margin * h  < point[1] < y + ((1.5 if margin>=0 else 1) + margin) * h :  # 框内,1.5是考虑把脖子排除掉
            inrect = True
            break
    return inrect

# RESTful接口定义
class myRestAPP(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

# 打开视频，主要解决自动屏蔽Windows平台下开摄像头出现警告信息的问题
def myVideoCapture(cameraID=0):
    # 如Windows下打开摄像头警告cap_msmf.cpp (435) `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback
    # 用cap = cv2.VideoCapture(cameraID, cv2.CAP_DSHOW)可以不出现警告
    if isinstance(cameraID,int):
        if platform.system()=='Windows':
            cap = cv2.VideoCapture(cameraID,cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cameraID)
    else:
        cap = cv2.VideoCapture(cameraID)
    return cap

# 计算循环队列queue中最近n个元素的众数，point为队列当前指针
def lastNMode(queue,point=0,n=None):
    mode=None # 众数
    qLen=len(queue)
    if qLen>0: # 队列非空才处理
        if n is None: # 未传入你，则取整个队列元素
            n = len(queue)
        elif n<0 or n >qLen: # n超出有效范围，也取整个队列元素
            n = len(queue)

        if point<0 or point>=qLen:#point超出范围，归0
            point=0

        if n<=point+1:# point之前的元素就够n个了
            n_queue = queue[point-n+1:point+1]
        else: # 需要拼接两段队列
            k = n-point-1# 前面可以得到point+1个元素，还需从队尾截取k个元素（放到新队手部）
            n_queue = queue[qLen-k:qLen]+queue[0:point + 1]
        mode = max(n_queue, key=n_queue.count)
    return mode

if __name__ == '__main__':
    pass