from matplotlib import pyplot as plt
from myGlobalVariables import *
import myGlobalVariables as g
from myCommonModules import *
import myKalmanFilter as kf
import cv2

# 分割手部图像
# 输入lastFrame,currentFrame分别为上一帧和当前帧彩色图像
# face_cascade为Haar人脸检测器，速度快（直接传入，避免重复加载）
# resNetFace为ResNet人脸检测器，准确率高
# dlibFace为Dlib Hog人脸检测器，准确率较高，速度较快
# HandAreaScope=[3000,102400]轮廓面积落在此区间，才作为候选手部处理
# useWaterShed=True,启用分水岭算法
# moveSeg=True，开启运动分割
# useBackSeg=True，开启背景分割
# 返回手部轮廓、掩膜和手部中心位置(x,y)
def segHand(lastFrame, currentFrame, faces=None,handAreaScope=[3000,102400],
            useWaterShed=True, moveSeg=True,useBackSeg=True):
    # 分割阈值参数设置
    # YCrCb中133 <= Cr <= 173, 77 <= Cb <= 127
    # HSV中, 7 < H < 20, 28 < S < 256, 50 < V < 256
    # 正常黄种人的Cr分量大约在140至175之间，Cb分量大约在100至120之间
    # hsvMinThreshold,hsvMaxThreshold =3 , 20 # h通道中位于此区间内的像素是黄色皮肤
    # labMinThreshold, labMaxThreshold = 136, 152  # b通道中位于此区间内的像素是黄色皮肤
    # YCrMinThreshold,YCrMaxThreshold=135,175 # YCrCb颜色空间Cr通道中位于此区间内的像素是黄色皮肤
    # YCbMinThreshold, YCbMaxThreshold = 80, 160  # YCrCb颜色空间Cb通道中位于此区间内的像素是黄色皮肤
    moveThreshold = 5  # 帧差法中色差超过此阈值的像素会被作为运动目标分割出来，设大了手部不完整，小了受光照不均匀影响大假目标多
    if len(handAreaScope)!=2:
        HandAreaScope=[3000,102400]  # 轮廓面积落在此区间才能被当做手检测

    # 帧差，提取色差超过一定阈值的像素，实现运动目标分割，背景有接近肤色颜色时需启用运动分割
    if moveSeg:
        moveHand = cv2.absdiff(currentFrame, lastFrame)  # 帧差
        moveHand = cv2.cvtColor(moveHand, cv2.COLOR_BGR2GRAY)  # 转灰度
        moveHand = cv2.inRange(moveHand, moveThreshold, 255)  # 运动分割

    # # hsv空间肤色分割，效果不太好，但可以考虑与ycrcb互补，暂未用
    # hsvFrame=cv2.cvtColor(currentFrame,cv2.COLOR_BGR2HSV )[:,:,0] # 转换到hsv空间后取h通道
    # hsvHand=cv2.inRange(hsvFrame,hsvMinThreshold,hsvMaxThreshold) # 肤色分割

    # 绘制h通道颜色直方图，用以确定肤色分割阈值
    # hist = cv2.calcHist([hsvFrame],[0], None, [256], [0, 255])
    # plt.plot(hist,color='blue')

    # # lab空间肤色分割,效果非常不好，噪声大，不用了
    # labFrame=cv2.cvtColor(currentFrame,cv2.COLOR_BGR2LAB )[:,:,2] # 转换到lab空间后取b通道
    # labHand = cv2.inRange(labFrame, labMinThreshold, labMaxThreshold) # 肤色分割

    # 绘制b通道颜色直方图，用以确定肤色分割阈值
    # hist = cv2.calcHist([labFrame],[0], None, [256], [0, 255])
    # plt.plot(hist,color='green')
    # plt.show()

    if useBackSeg:
        global frameBackGround  # 背景模型
        fgMask = frameBackGround.apply(currentFrame)  # 减除背景，得到前景（手部会不完整，手心处有空洞）
        # fgMask = polish(fgMask,small_kernel_size=5,big_kernel_size=20)

    # YCrCb颜色空间肤色分割，ycrcbFrame, YCrCb图像，ycrFrame, Cr通道图像，ycrcbHand，肤色区域掩膜
    # YCrMinThreshold = 135, YCrMaxThreshold = 175,YCbMinThreshold = 80, YCbMaxThreshold = 160 #常规肤色范围
    # YCrMinThreshold, YCrMaxThreshold,  YCbMinThreshold,  YCbMaxThreshold =  125,150,115,125 # 本程序常用范围
    YMinThreshold, YMaxThreshold, YCrMinThreshold, YCrMaxThreshold, YCbMinThreshold, YCbMaxThreshold = 38, 255, 133, 255, 103, 133
    ycrcbFrame, ycrFrame, ycrcbHand = YCrCbSeg(currentFrame, YMinThreshold=YMinThreshold, YMaxThreshold=YMaxThreshold,
                                               YCrMinThreshold=YCrMinThreshold, YCrMaxThreshold=YCrMaxThreshold,
                                               YCbMinThreshold=YCbMinThreshold, YCbMaxThreshold=YCbMaxThreshold,faces=faces)
    # ycrcbHand=polish(ycrcbHand,5,5)

    # 生成包含完整前景且肤色区的掩膜
    ycrFrameSeg = cv2.bitwise_and(ycrFrame, ycrFrame, mask=ycrcbHand)  # 把肤色区图像分割出来（手部一定包含其中，会有冗余的接近肤色的背景）
    if useBackSeg:
        fgYCrHand = cv2.bitwise_and(fgMask, ycrcbHand, mask=fgMask)  # 背景差分与肤色分割交叠区，排除冗余背景和非肤色区域，但手部有空洞
        fgYCrHand = polish(fgYCrHand, small_kernel_size=0, big_kernel_size=20)
    else:
        ycrcbHand=polish(ycrcbHand, small_kernel_size=5, big_kernel_size=20)
        fgMask=ycrcbHand
        fgYCrHand=ycrcbHand

    # # 斑点分析
    # params = cv2.SimpleBlobDetector_Params()
    # # 颜色过滤
    # params.blobColor=0 # 检测黑色斑点
    # # 面积过滤
    # params.filterByArea = True
    # params.minArea = 1
    # params.maxArea = 100000
    # # 斑点间最小距离
    # minDistBetweenBlobs = 1
    # holeDetector = cv2.SimpleBlobDetector_create(params) # 检测缺失的黑洞
    # holes = holeDetector.detect(fgYCrHand)
    # # if len(holes)>0: print("holes",holes[0].pt,holes[0].size)
    # for h in holes:
    #     cv2.circle(fgYCrHand,(int(h.pt[0]),int(h.pt[1])),int(h.size),127,-1)
    # im_with_keypoints = cv2.drawKeypoints(currentFrame, holes, np.array([]), (255, 255, 0),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # showImg(im_with_keypoints, 'im_with_keypoints', xScale=1)

    fgContours =myFindContours(fgYCrHand)  # 取交叠区轮廓，填充时取点用
    fgContours.sort(key=cv2.contourArea, reverse=True)  # 按轮廓面积降序排列


    # 调试用#######################################
    # test=0
    # for c in fgContours:
    #     if test==0:
    #         cv2.drawContours(currentFrame, [c], 0, (0, 255, 255), 1)
    #     else:
    #         cv2.drawContours(currentFrame,[c], 0, (255, 255, 255), 1)
    #     test+=1
    #     if test>5:
    #         break
    # showImg(fgMask,'fgMask',xScale=0.5)
    # showImg(ycrcbHand, 'ycrcbHand', xScale=0.5)
    # showImg(fgYCrHand, 'fgYCrHand2', xScale=0.5)
    ###############################################

    # 分水岭分割，主要目的是生成边界（轮廓只能圈出一些小圈，不能对图像构成分割），遏制洪泛填充时手部和背景粘连
    if useWaterShed:
        markers = WaterShed(ycrcbFrame, ycrcbHand)  # 分割速度慢
        ycrFrameSeg[markers == -1] = 255  # 边界变白，形成阻隔填充的边界
        if not faces is None:
            for (x, y, w, h) in faces:
                cv2.rectangle(ycrFrameSeg,(x,y),(x+w,y+h),255,1) # 把人脸框也当边界画进去

        # # 显示边界效果看看
        # currentFrame[markers == -1] = (255, 255, 0)
        # showImg(ycrFrameSeg, 'markers',xScale=1)

        fillMask = np.zeros([int(currentFrame.shape[0] + 2), int(currentFrame.shape[1] + 2)], np.uint8)  # 洪泛填充掩膜、待填充
        filled = 0  # 记录填充了几个点了
        for c in fgContours:  # 每个轮廓内取一个肤色像素作为种子，进行洪泛填充
            if len(c) < 10 or filled > 10 :  # 往后的轮廓已经很短了，或已填充10个了，就结束
                break
            # seed = getCenterXY(c) # 取轮廓中心作为种子
            # seed = tuple(c[0][0]) # 取轮廓上的一点作为种子

            if not inRect(faces,getCenterXY(c)):# and handAreaScope[0] <= cv2.contourArea(c) <= handAreaScope[1]: #轮廓中心不在脸部，且符合手部大小
                neighbor=20
                testPoints = [[neighbor, neighbor], [-neighbor, -neighbor], [-neighbor, neighbor], [neighbor, -neighbor],
                              [-neighbor, 0], [neighbor, 0], [0, -neighbor], [0, neighbor]]  # 8邻域
                for tP in testPoints:  # 取轮廓内一点作为种子
                    seed = tuple(c[0][0] + tP)  # 种子
                    if cv2.pointPolygonTest(c, seed, False) == 1 \
                            and YCrMinThreshold<=ycrFrameSeg[seed[1]][seed[0]]<=YCrMaxThreshold: # 所选点在轮廓内，且为肤色区域

                        # # 调试
                        # cv2.drawContours(currentFrame, [c], 0, (0, 0, 255), 1)
                        # cv2.circle(currentFrame,seed,5,(0,0,255),-1)
                        # cv2.putText(currentFrame,str(filled),seed, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                        # cv2.putText(currentFrame,str(cv2.contourArea(c)),seed, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)

                        histMask = np.zeros([currentFrame.shape[0], currentFrame.shape[1]],np.uint8)  # 绘制手部掩膜用
                        histMask = cv2.drawContours(histMask, [c], 0, 255, -1) # 绘制手部掩膜
                        # showImg(histMask,'histMask',xScale=0.5)
                        hist = cv2.calcHist([ycrFrameSeg], [0], histMask, [256], [0, 255])
                        modeColor=np.argmax(hist) # 颜色直方图的众数，即颜色中心
                        lowCr=int(max(YCrMinThreshold,modeColor-5)) # 手部Cr通道肤色下限
                        upCr=int(min(modeColor+5,YCrMaxThreshold)) # 手部Cr通道肤色上限
                        # mean, std = cv2.meanStdDev(ycrFrameSeg,mask=histMask) # 手部Cr通道颜色均值和标准差
                        # lowCr=int(max(0,mean-std)) # 手部Cr通道肤色下限
                        # upCr=int(min(mean+std,255)) # 手部Cr通道肤色上限

                        if ycrFrameSeg[seed[1]][seed[0]]>=lowCr:
                            lowDiffColor=int(ycrFrameSeg[seed[1]][seed[0]]-lowCr)
                        else:
                            lowDiffColor = 1
                        if ycrFrameSeg[seed[1]][seed[0]]<=upCr:
                            upDiffColor = int(upCr-ycrFrameSeg[seed[1]][seed[0]])
                        else:
                            upDiffColor = 1

                        # print(ycrFrameSeg[seed[1]][seed[0]],modeColor,lowDiffColor,upDiffColor)
                        cv2.floodFill(ycrFrameSeg, fillMask, seed, 255, lowDiffColor, upDiffColor,
                                      255 << 8 | cv2.FLOODFILL_MASK_ONLY)  # 与肤色区对应的掩膜区（相邻像素色差小于阈值）被填充为255
                        filled += 1
                        break

        fgMask = fillMask[1:currentFrame.shape[0] + 1, 1:currentFrame.shape[1] + 1]  # 前景且肤色区掩膜
        # showImg(fgMask, 'fgMask2', xScale=0.5)

    # # 显示YCrCb分割结果
    # showImg(ycrcbHand, 'YCrCbSegHand', xScale=0.8)

    # # 显示YCrCb抠图结果
    # showImg(ycrFrameSeg, 'ycrFrameSeg', xScale=0.8)
    #
    # # 显示背景差分与肤色分割交叠区
    # showImg(fgYCrHand, 'fgYCrHand', xScale=0.8)
    #
    # # 显示前景且肤色区掩膜
    # showImg(fgMask, 'fgMask', xScale=0.8)

    # # 显示运动分割结果
    # if moveSeg:
    #     showImg(moveHand, 'moveHand', xScale=0.8)
    #     showImg(cv2.bitwise_and(moveHand, fgMask), 'move and BK', xScale=0.8)

    # 分割结果融合
    if moveSeg:
        hand = cv2.bitwise_and(moveHand, ycrcbHand, mask=fgMask)
    else:  # 如果不用运动分割信息，手和脸粘连时，脸会被裹进来
        hand = cv2.bitwise_and(ycrcbHand, ycrcbHand, mask=fgMask)

    # 形态学操作（开操作+闭操作）去噪、填充缝隙
    # hand=polish(hand,5,5)

    global TrackingHand  # Kalman滤波器手部跟踪器
    # 给返回值设初值
    handContour = []  # 用于保存手部轮廓
    handContours = []  # 用于保存候选手部轮廓
    handMask = np.zeros((currentFrame.shape[0], currentFrame.shape[1]), dtype=np.uint8)  # 用于保存手部掩膜
    handPosition = (-1, -1)  # 用-1标识未分割到手部

    # 提取轮廓
    bincontours = myFindContours(hand)

    if len(bincontours) > 0:
        # maxContour = max(bincontours,key = cv2.contourArea) # 计算最大轮廓
        bincontours.sort(key=cv2.contourArea)  # 按轮廓面积升序排列
        lc = len(bincontours)
        for c in range(lc - 1, -1, -1):  # 从大到小遍历轮廓
            contour = bincontours[c]  # 取轮廓
            if handAreaScope[0] <= cv2.contourArea(contour) <=handAreaScope[1]:
                # 计算手部中心位置坐标
                handPosition = getCenterXY(contour)
                inFace = inRect(faces, handPosition, margin=0.2)  # 判断轮廓中心是否落在人脸框内部
                if not inFace:  # 如果检测到的轮廓不是人脸，就记下来备选
                    handContours.append((bincontours[c], handPosition))
                    # break # 找到一个手就结束遍历
            else:  # 轮廓太小了，就不继续往下遍历
                break

        # 检查每一个候选手，哪个最接近Kalman滤波器跟踪结果就选哪个作为手部
        # 如果Trackinghand对象尚未创建，就以handContours[0][0]作为手部
        if len(handContours) > 0:  # 有候选手
            handContour = handContours[0][0]  # 第0个首选手部
            if not TrackingHand is None:  # 如果建立了跟踪对象
                minTrackDist = currentFrame.shape[0] * currentFrame.shape[1]  # 记录最小跟踪距离
                for (hc, hp) in handContours:
                    LastDist = np.linalg.norm(np.array(hp) - np.array(TrackingHand.measurement))  # 当前手与上一帧手的距离
                    PredictDist = np.linalg.norm(np.array(hp) - np.array(TrackingHand.prediction))  # 当前手与预测手的距离
                    TrackDist = LastDist + PredictDist  # 综合考虑与上一帧手的距离和预测手的距离
                    if TrackDist < minTrackDist:  # 发现与上一帧手和预测手更近的候选手
                        minTrackDist = TrackDist
                        handContour = hc

        handPosition = (-1, -1)
        if len(handContour) > 0:
            # 用近似轮廓包络手部，减少边缘锯齿
            arcLenght = cv2.arcLength(handContour, True)
            handContour = cv2.approxPolyDP(handContour, min(arcLenght * 0.01, 3), True)
            # print('手部轮廓长度',cv2.arcLength(handContour,True))

            # 手臂去除
            handMask=removeArm(handContour,handMask,rFactor=1.4)

            # 重新生成手部轮廓
            handContour = myFindContours(handMask)
            if len(handContour)>0:
                handContour = max(handContour, key=cv2.contourArea)  # 计算最大轮廓

                # 手部位置
                handPosition = getCenterXY(handContour)

                # 检测到的手部画到原图中，会使检测结果更稳定。此前不要重新生成currentFrame，否则无法传递到lastFrame
                cv2.drawContours(currentFrame, [handContour], -1, (255, 0, 0), cv2.FILLED)

                # Kalman滤波跟踪手部
                if TrackingHand is None:
                    TrackingHand = kf.KalmanFilter(0, handPosition)
                TrackingHand.Correct(handPosition)
                predictPosition = TrackingHand.Predict()
                cv2.circle(currentFrame, handPosition, 2, (0, 255, 0), -1)  # 手部检测位置
                cv2.circle(currentFrame, (predictPosition), 2, (0, 0, 255), -1)  # 手部预测位置

    # 画人脸框（绿色框，边框宽度为2）
    for (x, y, w, h) in faces:
        currentFrame = cv2.rectangle(currentFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return handContour, handMask, handPosition

# 手臂消除
# 输入：
# handContour 手部轮廓（可能带手臂）
# handMask 空白画布
# rFactor=1.2，外割圆半径比例因子，外割圆半径 = rFactor * 内接圆半径
# 输出：
# handMask 去除手臂后的手部掩膜
def removeArm(handContour,handMask,rFactor=1.2):
    cutHand=np.copy(handMask)
    handMask = cv2.drawContours(handMask, [handContour], -1, 255, cv2.FILLED)

    if cv2.contourArea(handContour)>0: # 手部面积大于0才处理，避免除0错误
        rect = cv2.minAreaRect(handContour)  # 最小外界矩形，返回（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）,最低点右侧的边长为宽度
        w, h, wIdx, hIdx = cv2.minMaxLoc(rect[1])  # 外接矩形短边长、长边长、短边索引号、长边索引号
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # print('手部最小外接矩形高宽比：',h/w)
        if h/w>2.5: # 外接矩形细高时，截取上部2/3
            # 计算长边2/3点（上部2/3，下部1/3）
            if rect[1][0] == w:  # cv2.minAreaRect返回的宽度小于高度，短边在最低点右侧，短边左下角的顶点序号是0，底部短边端点是box[0]和box[3]
                leftPoint = box[1]+(box[0] - box[1]) * 2 / 3
                rightPoint = box[2]+(box[3] - box[2]) * 2 / 3
                newRectContour = np.array([[box[0]],[box[3]],[rightPoint],[leftPoint]])
            else:  # 短边右下角的顶点序号是0，底部短边端点是box[0]和box[1]
                leftPoint = box[2]+(box[1] - box[2]) * 2 / 3
                rightPoint =box[3]+ (box[0] - box[3]) * 2 / 3
                newRectContour = np.array([[box[1]], [leftPoint], [rightPoint], [box[0]]])
            handMask = cv2.drawContours(handMask, [newRectContour.astype(int)], 0, 0, -1)

        disHand = cv2.distanceTransform(handMask, 2, 3) # 距离变换

        # max_indx是距离图最大值位置，内接圆心，max_val是距离图最大值，内接圆半径
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(disHand)
        # print(min_val,max_val,min_indx,max_indx)

        # cv2.circle(disHand,max_indx,int(max_val),0,2) # 内接圆
        # cv2.circle(disHand, max_indx, int(max_val*rFactor), 0, 2)  # 外割圆
        # showImg(disHand,'dishand',xScale=0.5)

        # 计算短边中点
        if rect[1][0] == w:  # cv2.minAreaRect返回的宽度小于高度，短边在最低点右侧，短边左下角的顶点序号是0，底部短边端点是box[0]和box[3]
            middleTopPoint = (box[1] + box[2]) / 2
            middleBottomPoint = (box[0] + box[3]) / 2
        else: # 短边右下角的顶点序号是0，底部短边端点是box[0]和box[1]
            middleTopPoint = (box[2] + box[3]) / 2
            middleBottomPoint = (box[0] + box[1]) / 2

        # 按大圆与矩形短边（平移）切线截断
        p1, p2 = crossCircleLine(max_indx, int(max_val * rFactor), (middleTopPoint[0], middleTopPoint[1]), (middleBottomPoint[0], middleBottomPoint[1]))
        if not p1 is None and not p2 is None:
            pc0=list(p2 if p1[1] < p2[1] else p1) # 位于下方的那个交点的坐标
            if p1[0]-p2[0]==0: # 无斜率
                box[0][1] = pc0[1]
                if rect[1][0] == w: #底部短边端点是box[0]和box[3]
                    box[3][1] = pc0[1]
                else:# 底部短边端点是box[0]和box[1]
                    box[1][1] = pc0[1]
            else:#有斜率
                k=(p1[1]-p2[1])/(p1[0]-p2[0])# 中线斜率
                if k==0:
                    box[0][0]=pc0[0]
                    box[1][0]=pc0[0]
                else:
                    k2=-1/k # 新短边斜率
                    if rect[1][0] == w: #底部短边端点是box[0]和box[3]
                        pc1 = crossLine(box[0], box[1], k2, pc0)  # 计算矩形左侧长边截断位置
                        pc2 = crossLine(box[2], box[3], k2, pc0)  # 计算矩形左侧长边截断位置
                        box[0] = list(pc1)
                        box[3] = list(pc2)
                    else:# 底部短边端点是box[0]和box[1]
                        pc1 = crossLine(box[1], box[2], k2, pc0)  # 计算矩形左侧长边截断位置
                        pc2 = crossLine(box[0], box[3], k2, pc0)  # 计算矩形左侧长边截断位置
                        box[1] = list(pc1)
                        box[0] = list(pc2)

        # # for i in range(len(box)): # 显示顶点序号
        # #     cv2.putText(disHand,str(i), (box[i][0], box[i][1]), cv2.FONT_HERSHEY_PLAIN, 2, 255, thickness=2)
        # # 按大圆与矩形框下部交点截断
        # if rect[1][0] == w: # 左下角的顶点序号是0
        #     # 求外接矩形与外割圆的交点，取下面的两个点做手腕分割线
        #     p1, p2 = crossCircleLine(max_indx, int(max_val * rFactor), (box[0][0], box[0][1]), (box[1][0], box[1][1]))
        #     p3, p4 = crossCircleLine(max_indx, int(max_val * rFactor), (box[3][0], box[3][1]), (box[2][0], box[2][1]))
        #     if not p2 is None and not p4 is None:
        #         # 用交点代替外接矩形顶点（排除新矩形之外的内容）
        #         box[0] = list(p2 if p1[1] < p2[1] else p1)
        #         box[3] = list(p4 if p3[1] < p4[1] else p3)
        # else: # 右下角的顶点序号是0
        #     p1, p2 = crossCircleLine(max_indx, int(max_val * rFactor), (box[1][0], box[1][1]), (box[2][0], box[2][1]))
        #     p3, p4 = crossCircleLine(max_indx, int(max_val * rFactor), (box[0][0], box[0][1]), (box[3][0], box[3][1]))
        #     if not p2 is None and not p4 is None:
        #         box[1] = list(p2 if p1[1] < p2[1] else p1)
        #         box[0] = list(p4 if p3[1] < p4[1] else p3)

        cv2.drawContours(cutHand, [box], 0, 255, cv2.FILLED)  # 画外界矩形做mask
        handMask = cv2.bitwise_and(handMask, handMask, mask=cutHand) # 只保留手部
        # cv2.imshow('RemoveArm', handMask)
    return handMask