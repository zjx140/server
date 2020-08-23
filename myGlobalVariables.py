import cv2
import numpy as np
import time, os
from myConfig import *

# # 手形定义字典，每个三元组[手部面积占比范围，手部最小外接图像宽高比范围，手形名称]定义一个手形
# handShapeDict = [[[0.7, 0.8], [0.3, 0.6], "palm"],
#                  [[0.5, 0.7], [0.4, 0.6], "finger"]]

# 手形定义字典，每个4元组[手部面积占比范围，手部最小外接图像宽高比范围，手指手掌面积占比范围，手形名称]定义一个手形
handShapeDict = [[[0.5, 0.7], [0.4, 0.6], [0.01, 0.1], "finger"],
                 [[0.7, 0.8], [0.3, 0.6], [0.1, 0.7], "palm"]]

# 手势定义字典，每个三元组[手形，运动方向，手势名称]定义一个手势
handGesDict = [['finger', 'left', 'moveleft'],
               ['finger', 'right', 'moveright'],
               ['finger', 'up', 'moveup'],
               ['finger', 'down', 'movedown'],
               ['finger', 'clockwise', 'zoomin'],
               ['finger', 'anticlockwise', 'zoomout'],
               ['palm', 'left', 'turnleft'],
               ['palm', 'right', 'turnright'],
               ['palm', 'up', 'close'],
               ['palm', 'down', 'return'],
               ['palm', 'clockwise', 'ok'],
               ['palm', 'anticlockwise', 'cancel']]

# 记录手形识别数据[手部面积占比范围，手部最小外接图像宽高比范围，手形名称]
handdata = []

# videoProcessing函数中用
handTrackLen = 15  # 跟踪的手部运动轨迹长度
handTrack = list([(0, 0)] * handTrackLen)  # 记录手部轨迹坐标元组的循环列表
hPoint = 0  # handTrack列表当前位置指针
conHandTrackLen = 8  # 用连续conHandTrackLen次轨迹判定结果生成最终轨迹，影响灵敏度，不大于handTrackLen
conHandTrack = list(['static'] * conHandTrackLen)  # 记录手部轨迹识别结果，循环列表
handShapes = list([None] *  conHandTrackLen)  # 记录手形的循环列表，用于平滑手形识别结果
tPoint = 0  # conHandTrack列表当前位置指针

# 背景建模，用于手部分割，history设小了，停顿的手会成为背景，影响检测
frameBackGround = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False)

TrackingHand = None  # 正在跟踪的手部对象,Kalman滤波

# 调参、调试用=====================================
myTrackBar = None # 调节工具条，主要供调节颜色参数用
myPlot=None # 绘图工具，主要供查看颜色直方图调参用
# ================================================
