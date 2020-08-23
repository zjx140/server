from myGlobalVariables import *
from sklearn.ensemble import IsolationForest

# 手部运动识别
# 输入手部轨迹列表handTrack，该列表形如[(x1,y1),(x2,y2),...]，连续记录过去若干帧的手部位置
# hand是可选输入参数，主要是调试时用于在画面绘制一些图形用，使程序计算结果更直观，便于调试
# 返回手部运动方向(left, right,up, down, clockwise, anticlockwise, static)及置信度
def handMovRecog(handTrack, hand=None):
    # 测试用轨迹数据，17点，第2、14、16号数据为离群点
    # handTrack = [(325, 306), (360, 314), (567, 426), (367, 326), (352, 330), (333, 326), (328, 317), (327, 305),
    #              (328, 289), (343, 280), (362, 274), (380, 273), (398, 271), (400, 272), (534, 215), (303, 303),
    #              (-1, -1)]
    # print('handTrack:', len(handTrack), handTrack)

    handMovement = "unkown"  # 根据轨迹判断结果，用运动方向名称给此变量赋值
    confidence = 1  # 根据轨迹判断结果的自信程度设置此值
    maxDisError = 100  # 容许的两点之间各维度最大距离误差，用于清理离群点
    minRadius = 20  # 圆周运动最小识别半径
    minDisT = 50  # 容许的轨迹点间最大距离的最小平均值，超过此平均值阈值才视为运动
    maxDist = 100  # 直线运动时，与运动方向垂直方向上的波动范围不能超过此值
    global TrackingHand  # 手部跟踪器

    handTrack = [(x, y) for x, y in handTrack if x > 0]  # 滤掉初始值(0,0)和未分割到手势值(-1,-1)

    if len(handTrack) > 0:
        # 利用iForest算法计算轨迹中的离群点
        iForest = IsolationForest(n_estimators=10)
        iForest.fit(handTrack)
        Outliers = iForest.predict(handTrack)  # 1为正常值，-1为异常值
        # print(Outliers)
        # handTrackFiltered=[handTrack[i] for i in range(len(handTrack)) if Outliers[i] == 1] # 滤掉异常值

        if Outliers[0] == 1:  # 处理起点，不是离群点就留下
            handTrackFiltered = [handTrack[0]]
        else:  # 是离群点就舍弃
            handTrackFiltered = []

        # 再次检测离群点，算法算出的离群点不都准，偏离未超过maxDisError的“离群点”也作为正常点保留。未考虑子序列离群。
        for i in range(1, len(handTrack) - 1):
            if Outliers[i] == -1:
                d1 = (handTrack[i][0] - handTrack[i - 1][0]) ** 2 + (
                            handTrack[i][1] - handTrack[i - 1][1]) ** 2  # 与前一点的平方距离
                d2 = (handTrack[i][0] - handTrack[i + 1][0]) ** 2 + (
                            handTrack[i][1] - handTrack[i + 1][1]) ** 2  # 与后一点的平方距离
                if d1 < maxDisError ** 2 and d2 < maxDisError ** 2:  # 偏差
                    Outliers[i] = 1  # 偏差不大的点重新置为正常点
                    handTrackFiltered.append(handTrack[i])
            else:
                handTrackFiltered.append(handTrack[i])

        if Outliers[-1] == 1:  # 处理终点，不是离群点就留下
            handTrackFiltered.append(handTrack[-1])

        # print('handTrackFiltered:', len(handTrackFiltered), handTrackFiltered)

        if len(handTrackFiltered) > 1:
            # 计算有效轨迹点之间最大距离
            minXY = np.min(handTrackFiltered, axis=0)
            maxXY = np.max(handTrackFiltered, axis=0)
            dXY = maxXY - minXY
            # print(dXY)

            # 计算相邻点间各维度差
            dxdy = []
            for i in range(1, len(handTrackFiltered)):
                dx = handTrackFiltered[i][0] - handTrackFiltered[i - 1][0]
                dy = handTrackFiltered[i][1] - handTrackFiltered[i - 1][1]
                dxdy.append((dx, dy))
            dxdymean = np.mean(dxdy, axis=0)  # 计算均值，用于判断方向
            # dxdyabsmean = np.mean(np.abs(dxdy), axis=0) # 计算绝对值均值，用于判断偏差

            TrackArea = cv2.contourArea(np.array(handTrackFiltered))  # 轨迹面积
            TrackPerimeter = cv2.arcLength(np.array(handTrackFiltered), True)  # 轨迹周长
            # TrackRadius = TrackPerimeter/np.pi/2 # 估算圆形轨迹半径
            # 圆形度=（4π * 面积） / （周长 * 周长）
            TrackCircularity = 0.0
            if TrackPerimeter > 0:
                TrackCircularity = 4 * np.pi * TrackArea / (TrackPerimeter * TrackPerimeter)
            # print('圆度：',TrackCircularity,'水平跨度：',dXY[0],'垂直跨度：',dXY[1],'方向：',dxdymean)

            if dXY[0] > minRadius * 2 and dXY[1] > minRadius * 2 and TrackCircularity > 0.4:  # 纵横运动距离均大于半径阈值的2倍，可能为圆形运动
                # 判断旋转方向的思路：在图像坐标系中（y轴向下），顺时针旋转，方位角（“轨迹圆心-手”向量与x轴夹角）增大
                # 统计轨迹点坐标均值可以计算出轨迹圆的中心，利用反正切函数可以算出方位角（要折算到0-2pi之间，过x正半轴需做特殊考虑）
                # 统计方位角增量，可以判定顺时针还是逆时针
                circleCenter = np.mean(handTrackFiltered, axis=0)  # 计算圆形运动轨迹中心
                if hand is not None:  # 如果可选参数传入了hand，就画出来中间计算结果
                    cv2.circle(hand, (int(circleCenter[0]), int(circleCenter[1])), 10, 255, -1)
                    cv2.line(hand, (int(circleCenter[0]), int(circleCenter[1])),
                             (handTrackFiltered[len(handTrackFiltered) - 1]), 255, 1)
                lastAngle = 10  # 记录上一个轨迹点方位角，随便设置一个2pi范围以外的初值
                DeltaAngle = []  # 记录手挥过的角度差
                for handPos in handTrackFiltered:
                    aTan = np.arctan2(handPos[1] - circleCenter[1], handPos[0] - circleCenter[0])  # 计算方位角
                    if aTan < 0:  # 转换到0-2pi之间
                        aTan = aTan + np.pi * 2
                    if lastAngle != 10:  # 如果不是初值，那就是赋过值
                        if (aTan < np.pi / 2 and lastAngle > np.pi * 1.5):
                            DeltaAngle.append(aTan)  # 计算方位角增量，顺时针转回一圈时需做特殊判断
                        elif (lastAngle < np.pi / 2 and aTan > np.pi * 1.5):
                            DeltaAngle.append(0 - lastAngle)  # 计算方位角增量，逆时针转回一圈时需做特殊判断
                        else:
                            DeltaAngle.append(aTan - lastAngle)
                    lastAngle = aTan
                if len(DeltaAngle)>0:
                    dAngleMean = np.mean(DeltaAngle)  # 统计方位角增量正负
                else:
                    dAngleMean=0
                if dAngleMean < 0:
                    handMovement = "anticlockwise"
                else:
                    handMovement = "clockwise"
                confidence = 1 - TrackCircularity * 0.1
            elif dXY[0] > dXY[1] and dXY[0] > minDisT and dXY[1] < maxDist:  # 横向运动幅度大于纵向且大于运动阈值，纵向运动小于波动阈值
                if not TrackingHand is None:  # 利用Kalman滤波器预测位置与当前位置关系判断运动方向
                    if TrackingHand.prediction[0] > TrackingHand.measurement[0]:
                        handMovement = "right"
                    elif TrackingHand.prediction[0] < TrackingHand.measurement[0]:
                        handMovement = "left"
                else:  # 旧方法
                    if dxdymean[0] > 0:  # 向x轴正向运动
                        handMovement = "right"
                    elif dxdymean[0] < 0:  # 向x轴负向运动
                        handMovement = "left"
                confidence = 1 - dXY[1] / dXY[0] * 0.1
            elif dXY[1] > dXY[0] and dXY[1] > minDisT and dXY[0] < maxDist:  # 纵向运动幅度大于横向且大于运动阈值，横向运动小于波动阈值
                if not TrackingHand is None:  # 利用Kalman滤波器预测位置与当前位置关系判断运动方向
                    if TrackingHand.prediction[1] > TrackingHand.measurement[1]:
                        handMovement = "down"
                    elif TrackingHand.prediction[1] < TrackingHand.measurement[1]:
                        handMovement = "up"
                else:  # 旧方法
                    if dxdymean[1] > 0:  # 向y轴正向运动
                        handMovement = "down"
                    elif dxdymean[1] < 0:  # 向y轴负向运动
                        handMovement = "up"
                confidence = 1 - dXY[0] / dXY[1] * 0.1
            elif dXY[1] < minDisT and dXY[0] < minDisT:  # 横向纵向偏差均小于运动阈值
                handMovement = "static"
                confidence = 1 - (dXY[0] / minDisT) * (dXY[0] / minDisT) * 0.1
        # print(handMovement)
    return handMovement, confidence