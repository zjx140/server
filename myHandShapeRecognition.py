from myGlobalVariables import *
from myCommonModules import *
import myFourierDescriptor as fd


# 手形识别
# 输入手部轮廓handContour
# handsvm传入训练好的手形识别svm模型
# hand是可选输入参数，二值图像，白色为手部，主要是调试时用于在画面绘制一些图形用，使程序计算结果更直观，便于调试
# collectHandData=None不采集手形数据，如需采集手形数据，传入手形名称。
# 在当前目录collectHandData + r'_imgs/'中保存collectHandData + '_' + str(t) + '.jpg'手部图像
# 特征数据保存在全局变量handdata中，退出时保存到当前目录下collectHandData + '_data.txt'文件中
# 所采集数据供SVM训练用
# 返回手部静态姿势识别结果（finger, palm, other）及置信度
def handShapeRecog(handContour, handsvm=None, hand=None, collectHandData=None):
    # 输入的handContour有时检测不准，可能是面部，算法对此要有容错能力
    handShape = "other"  # 根据手形识别结果给此变量赋值
    confidence = 0.96  # 根据手势识别结果的自信程度设置此值，此处设置SVM的准确率
    if len(handContour) > 1:
        global handShapeDict
        handArea = cv2.contourArea(handContour)  # 手部面积
        if handArea>0:
            # 手部最小外接矩形，返回一个Box2D结构rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
            rect = cv2.minAreaRect(handContour)
            w, h, wIdx, hIdx = cv2.minMaxLoc(rect[1])  # 外接矩形短边长、长边长、短边索引号、长边索引号
            rectArea = w * h  # 手部最小外界矩形面积
            arcLength = cv2.arcLength(handContour, True)  # 手部周长
            hull = cv2.convexHull(handContour)
            # approx = cv2.approxPolyDP(handContour, max(w * 0.4, 10), True)
            # approx = cv2.approxPolyDP(handContour, w * 0.4, True)
            approx = cv2.approxPolyDP(handContour, min(arcLength * 0.01, 10), True)

            handAreaRatio = handArea / rectArea  # 手部面积占比
            AspectRatio = w / h  # 宽高比
            nApprox = len(approx)  # 近似轮廓点数
            nHull = len(hull)  # 凸包点数
            AreaLenghtRatio = rectArea / arcLength  # 手部面积周长比

            # valid, fDesciptor = fd.fourierDesciptor(handContour,descLen=8) # 傅里叶描述子

            # 新加的-------------------
            hand_center = cv2.distanceTransform(hand, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            _, max_rad, _, hand_position = cv2.minMaxLoc(hand_center)
            hand_copy = hand.copy()  # 复制一下，直接在上面画圆
            cv2.circle(hand_copy, hand_position, int(max_rad * 1.70), 0, -1)
            hand_copy_contours = myFindContours(hand_copy)
            contour_area = 0.0
            for hand_copy_contour in hand_copy_contours:
                contour_area += cv2.contourArea(hand_copy_contour)
            fingerandhandAreaRatio = contour_area / handArea  # 手指与手部占比
            # -------------------------

            # handsvm=None # 暂时屏蔽SVM，直接使用决策树算法

            # if handsvm is not None:  # 如果传入了svm模型，就用svm识别
            #     y = handsvm.predict([[handAreaRatio, AspectRatio]])  # 仅用面积占比和宽高比特征
            #     # y = handsvm.predict([[handAreaRatio, AspectRatio, nApprox, nHull, AreaLenghtRatio]])
            #     handShapeNameDict = {0: 'finger', 1: 'palm', 2: 'other'}
            #     handShape = handShapeNameDict[int(y[0])]
            # else: # 决策树
            #     for hs in handShapeDict:
            #         if hs[0][0] < handAreaRatio <= hs[0][1] and hs[1][0] < AspectRatio <= hs[1][1]:
            #             handShape = hs[2]
            #             break

            if handsvm is not None:  # 如果传入了svm模型，就用svm识别
                y = handsvm.predict([[handAreaRatio, AspectRatio]])  # 仅用面积占比和宽高比特征
                # y = handsvm.predict([[handAreaRatio, AspectRatio, nApprox, nHull, AreaLenghtRatio]])
                handShapeNameDict = {0: 'finger', 1: 'palm', 2: 'other'}
                handShape = handShapeNameDict[int(y[0])]
            else:
                for hs in handShapeDict:
                    if hs[0][0] < handAreaRatio <= hs[0][1] and hs[1][0] < AspectRatio <= hs[1][1]:
                        handShape = hs[3]
                        break
                if handShape == "other":
                    for rhs in handShapeDict:
                        if rhs[2][0] < fingerandhandAreaRatio <= rhs[2][1]:
                            handShape = rhs[3]
                            break

            fingers = countFingers(handContour)  # 手指计数
            # if fingers in ['1', '2'] and handShape == 'other':  # 融合手形识别与手指计数结果
            # # if fingers in ['1', '2'] and handShape == 'palm':  # 融合手形识别与手指计数结果
            #     txt='fingers_'+fingers+ 'handShape_'+handShape+'Final_finger'
            #     print(txt)
            #     t=str(time.time())
            #     filename=t+'_'+txt+'.jpg'
            #     print(filename)
            #     cv2.imwrite(filename, hand)
            #     handShape = 'finger'


            if hand is not None:
                cv2.putText(hand, "handArea%: " + str(np.round(handAreaRatio, 2)), (5, 40), cv2.FONT_HERSHEY_PLAIN, 1.6,
                            255, thickness=1)
                cv2.putText(hand, "AspectRatio: " + str(np.round(AspectRatio, 2)), (300, 40), cv2.FONT_HERSHEY_PLAIN, 1.6,
                            255, thickness=1)
                cv2.putText(hand, "nApprox: " + str(nApprox), (5, 60), cv2.FONT_HERSHEY_PLAIN, 1.6, 255, thickness=1)
                cv2.putText(hand, "ConvexHull: " + str(nHull), (300, 60), cv2.FONT_HERSHEY_PLAIN, 1.6, 255, thickness=1)
                cv2.putText(hand, "ArcLength: " + str(np.round(arcLength, 2)), (5, 80), cv2.FONT_HERSHEY_PLAIN,
                            1.6, 255, thickness=1)
                cv2.putText(hand, "AreaLenghtRatio: " + str(np.round(AreaLenghtRatio, 2)), (300, 80), cv2.FONT_HERSHEY_PLAIN,
                            1.6,255, thickness=1)
                cv2.putText(hand, "FingerHandRatio" + str(np.round(fingerandhandAreaRatio, 2)), (5, 100),
                            cv2.FONT_HERSHEY_PLAIN,1.6, 255, thickness=1)
                if fingers: cv2.putText(hand, "Fingers: " + fingers, (300, 100), cv2.FONT_HERSHEY_PLAIN, 1.6, 255, thickness=1)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(hand, [box], 0, 220)  # 画外界矩形
                cv2.polylines(hand, [approx], True, 180, 2) # 绘制近似轮廓
                # cv2.fillPoly(hand, [approx], 255)  # 绘制近似轮廓并填充
                cv2.drawContours(hand, [handContour], 0, 200, -1)
                # length = len(hull)
                # for i in range(len(hull)): # 绘制凸包
                #     cv2.line(hand, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), 200, 2)

                # 记录手形数据
                if collectHandData:
                    global handdata
                    t = time.time()  # 时间戳
                    # # 记录时间戳、面积率、宽高比、近似顶点数、凸包顶点数、面积周长比、手形名称
                    # handdata.append([t, handAreaRatio, AspectRatio, nApprox, nHull, AreaLenghtRatio, collectHandData])
                    # 记录时间戳、傅里叶描述子、手形名称
                    valid, fDesciptor = fd.fourierDesciptor(handContour, descLen=8)  # 傅里叶描述子
                    if valid:
                        handdata.append([t, list(fDesciptor), collectHandData])
                    cv2.imwrite(collectHandData + r'_imgs/' + collectHandData + '_' + str(t) + '.jpg', hand)
    return handShape, confidence


# 基于深度学习的手形识别
# def handShapeRecogbyDNN(currentframe, hand_model):
#     handShape = 'other'
#     confidence = 0.96
#     handPosition = (-1, -1)
#     handKeyPoints = hand_model.getHandKeyPoints(currentframe) #手部关键点
#     handShapes = hand_model.gesture(handKeyPoints) #手形list
#     colorhand = hand_model.visKeyPoints(np.copy(currentframe), handKeyPoints) #识别后的可视化图像（关键点连线）
#     if len(handShapes) == 0: #手形数组没有数据直接返回
#         return handShape, confidence, handPosition, colorhand
#     else: #若存在手形则等于第一个元素，若含有多个每加一个置信度减少0.01
#         handShape = handShapes[0]
#         confidence = 1 - (len(handShapes) - 1) * 0.01
#     handPosition = handKeyPoints[0] #首部位置用0号关键点表示
#     if handPosition is None: #若为0号为None则改为所有关键点的平均值
#         sum_a = 0
#         sum_b = 0
#         k = 0
#         for i in range(len(handKeyPoints)):
#             if handKeyPoints[i] is None:
#                 continue
#             else:
#                 k += 1
#                 a, b = handKeyPoints[i]
#                 sum_a += a
#                 sum_b += b
#         if k != 0:
#             handPosition = (sum_a / k, sum_b / k)
#     return handShape, confidence, handPosition, colorhand

# def handShapeRecogbyDNN(currentframe, hand_model):
#     handShape = ""
#     confidence = 0.96
#     # handPosition = (-1, -1)
#     handKeyPoints = hand_model.getHandKeyPoints(currentframe) #手部关键点
#     handShapes = hand_model.gesture(handKeyPoints) #手形list
#     #colorhand = hand_model.visKeyPoints(np.copy(currentframe), handKeyPoints) #识别后的可视化图像（关键点连线）
#     if len(handShapes) == 0: #手形数组没有数据直接返回
#         return handShape, confidence #, colorhand
#     else: #若存在手形则等于第一个元素，若含有多个每加一个置信度减少0.01
#         handShape = handShapes[0]
#         confidence = 1 - (len(handShapes) - 1) * 0.01
#     return handShape, confidence #, colorhand

# 手指计数
# 输入：
# cnt，手部轮廓
# currentFrame, 传入图像，用于调试时可视化
# 输出：result，手指计数
def countFingers(cnt,currentFrame=None):
    result=None
    try:
        # currentFrame=None # 不显示窗口
        frame=None
        if not currentFrame is None:
            if len(currentFrame.shape) == 3:# 彩色图
                frame = np.copy(currentFrame)
            else:#不是彩色图，转成彩色图
                frame = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2BGR)
        if len(cnt)>1:
            # 计算凸包
            hull = cv2.convexHull(cnt)
            if len(hull)>3:
                # 近似轮廓
                epsilon = 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # 计算凸包和手部轮廓面积
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)

                # 手部凸包未覆盖区域占手部轮廓覆盖区域面积比
                arearatio = ((areahull - areacnt) / areacnt) * 100

                # 凸性检测
                hull = cv2.convexHull(approx, returnPoints=False)#这里必须是False，返回点在轮廓中的索引
                defects = cv2.convexityDefects(approx, hull)
                if not defects is None:
                    l = 0 # 缺陷序号
                    # 计算缺陷
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0] # 起点、终点、远点在approx中的序号，远点到凸包最短距离(定点数）
                        start = tuple(approx[s][0]) # 起点（凸包顶点）坐标
                        end = tuple(approx[e][0]) # 终点（凸包顶点）坐标
                        far = tuple(approx[f][0]) # 远点（凹陷点）坐标
                        d = d/256 # 距离，顶点数换算成浮点数需除以256

                        # 计算三角形边长
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) # 起终点边长
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2) # 起点到凹陷点边长
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2) # 终点到凹陷点边长
                        # s = (a + b + c) / 2
                        # ar = math.sqrt(s * (s - a) * (s - b) * (s - c)) # 三角形面积
                        # d = (2 * ar) / a # 三角形a边上的高，也是凹点到凸包边线的距离，与转成浮点的d相同

                        # 计算凹陷点夹角
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                        # 忽略大于90度的角和距离凸包很近的点（一般为噪声）
                        if angle <= 90 and d > 30:
                            l += 1
                            if not frame is None: # 画凸包顶点
                                cv2.circle(frame, far, 5, [0, 0, 255], -1)
                        if not frame is None:# 画凸包线
                            cv2.line(frame, start, end, [0, 255, 0], 2)
                    l += 1

                    # 计算识别结果
                    if l == 1:
                        if areacnt < 2000: # 面积太小，不是手
                            result=None
                        else:
                            if arearatio < 12:
                                result='0'
                            else:
                                result='1'
                    elif l == 2:
                        result='2'
                    elif l == 3:
                        if arearatio < 27:
                            result='3'
                        else:
                            result='OK'
                    elif l == 4:
                        result='4'
                    elif l == 5:
                        result='5'

                    if not frame is None:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, result, (50, 50), font, 1, (0,0,255), 2)
                        cv2.imshow('frame', frame)
    except Exception as e:
        pass
    return result