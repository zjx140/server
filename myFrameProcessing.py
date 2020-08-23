from myConfig import *
from myGlobalVariables import *
from myCommonModules import *
from myHandSegment import segHand # 手部分割
from myHandMoveRecognition import handMovRecog # 手部运动轨迹识别
from myHandShapeRecognition import * # 手形识别
from myDynamicGesture import gestureRecog # 动态手势识别
import cv2

# 视频帧图像分析，currentFrame为当前帧图像，lastFrame为上一帧图像
# handsvm为手形识别SVM模型
# handdnn为手形识别DNN模型
# myFace人脸检测模型
# handAreaScope 手部面积范围
# useWaterShed=True,手部分割时，启用分水岭算法
# moveSeg=True,启用运动分割检测手部
# showVideo=True显示视频
# 如果保存视频，saveVideo传入vid_writer对象
# collectHandData为采集到的手形数据
# collectHandData=None不采集手形数据，如需采集手形数据，传入手形名称。所采集数据供SVM训练用
# 返回值：
# gesture,cg 手势识别结果和置信度
# lastFrame，将画上了人手的当前视频返回，为下一次处理用作lastFrame
def frameProcess(currentFrame, lastFrame, handsvm=None, handdnn=None,myFace=None,handAreaScope=[3000,102400],
                 useWaterShed=True, moveSeg=True, useBackSeg=True,showVideo=True,saveVideo=None, collectHandData=None):
    global handTrackLen  # 跟踪的手部运动轨迹长度
    global handTrack  # 记录手部轨迹坐标元组的循环列表
    global hPoint  # handTrack列表当前位置指针
    global conHandTrackLen  # 用连续conHandTrackLen次轨迹判定结果生成最终轨迹
    global conHandTrack  # 记录手部轨迹识别结果，循环列表
    global tPoint  # conHandTrack列表当前位置指针
    global handShapes# 手形识别结果列表

    t = time.time()

    # 人脸检测，faces为若干个人脸框[(x, y, w, h),(x, y, w, h),...]
    faces=myFace.detection(currentFrame)

    # # 高斯滤波
    # currentFrame = cv2.GaussianBlur(currentFrame, (7, 7), 0)
    originalFrame = currentFrame.copy()
    # 分割出手部轮廓handcontour,手部掩膜hand（二值图像，白色为手部），并算出手部位置handPosition
    # 如果handPosition=(-1,-1)，未分割到手势
    handContour, hand, handPosition = segHand(lastFrame, currentFrame, faces=faces,handAreaScope=handAreaScope,
                                              useWaterShed=useWaterShed,moveSeg=moveSeg,useBackSeg=useBackSeg)
    handShape=None
    # 深度学习手形识别
    if handdnn is not None:
        handShapeDNN,handBox,score=handdnn.predict(originalFrame)
        # print("handShapeDNN:", handShapeDNN, "handBox:",handBox, "score:", score)
        if handShapeDNN is not None:
            handShape=handShapeDNN
            ch=score
            y1, x1, y2, x2 = handBox
            handPosition=(int((x1+x2)/2),int((y1+y2)/2))
            # print("y1:",y1, "x1:", x1, "y2:",y2, "x2:", x2)
            # cv2.putText(currentFrame, dnnHandShape,(x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(currentFrame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    if handShape is None:
        # 识别手形handShape，置信度ch
        handShape, ch = handShapeRecog(handContour, handsvm, hand, collectHandData)

    handShapes[tPoint]=handShape

    # 用循环列表计算手部轨迹
    handTrack[hPoint] = handPosition
    hPoint = (hPoint + 1) % handTrackLen

    for p in handTrack:  # 绘制手部轨迹
        cv2.circle(currentFrame, p, 2, (0, 255, 0), -1)

    # 如果当前动作不是finger或palm，则统计历史识别结果的众数手形识别结果
    # print('本次手形识别结果：',handShape)
    # print('手形列表：',handShapes)
    if handShape not in ('finger','palm'):
        handShape = max(handShapes, key=handShapes.count)
    else: # 当前动作是finger或palm，考虑前面n个动作，平滑误识跳变
        handShape =lastNMode(handShapes,point=tPoint,n=3)
    # print('平滑后手形识别结果：',handShape)

    # 识别手部运动handMovement，置信度cm
    handMovement, cm = handMovRecog(handTrack[hPoint + 1:handTrackLen] + handTrack[0:hPoint + 1], hand)
    conHandTrack[tPoint] = handMovement
    tPoint = (tPoint + 1) % conHandTrackLen

    # 如果当前动作或上一动作是static或unkown，则不做任何容错处理，否则，统计历史识别结果的众数手部运动识别结果
    if handMovement not in ('static', 'unkown') and \
            conHandTrack[conHandTrackLen-1 if (tPoint - 1 == -1) else (tPoint - 1)] not in ('static', 'unkown'):
        handMovement = max(conHandTrack, key=conHandTrack.count)

    # 识别手势gesture，置信度cg
    gesture, cg = gestureRecog(handShape, handMovement, ch, cm)
    if gesture == 'invalid': cg = 1
    cg = np.round(cg, 2)

    # # 采集数据Start========================================
    # global handdata
    # t=time.time()
    # handdata.append([t, handShape, handMovement, gesture])
    # filepath=r'../dataset/1/'
    # text_save(handdata, filepath+r'video_labes.txt')
    # cv2.imwrite(filepath+r'frames/frame_' + str(t) + '_'+(handShape if not handShape is None else 'None')
    #             +'_'+(handMovement if not handMovement is None else 'None')
    #             +'_'+(gesture if not gesture is None else 'None')+'.jpg', currentFrame)
    # # 采集数据End==========================================

    # 计算处理速度（平均每帧处理速度）

    v = np.round((time.time() - t) / (handTrackLen), 5) * 1000
    v = str(v)[0:4] + ' ms'

    # 如果要显示或保存处理结果视频
    if showVideo or saveVideo:
        colorhand = cv2.merge([hand, hand, hand])  # 单通道图生成3通道图
        for p in handTrack:  # 绘制手部轨迹
            cv2.circle(colorhand, p, 2, (0, 255, 0), -1)
        hand = np.concatenate((currentFrame, colorhand), axis=1)  # 横向拼接视频

        # 在视频上显示识别结果和性能指标
        cv2.putText(hand, handShape, (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        cv2.putText(hand, handMovement, (150, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        if gesture == 'invalid':
            pass
            # cv2.putText(hand, gesture, (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), thickness=2)
        else:
            cv2.putText(hand, gesture, (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        cv2.putText(hand, "Time: " + v, (5, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness=2)
        cv2.putText(hand, "Confidence: " + str(cg), (400, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness=2)

        # 显示视频
        if showVideo:
            if VideoWinScale is not None:
                showImg(hand,'HandGesture',xScale=VideoWinScale,showOnTop=ShowOnTop)
            else:
                showImg(hand,'HandGesture',xScale=1,showOnTop=ShowOnTop)
        # 保存视频
        if saveVideo:
            saveVideo.write(hand)

    lastFrame = np.copy(currentFrame)  # 帧差分计算用

    return gesture, cg, lastFrame


# 基于深度学习手形识别的帧处理
# hand_model：手部识别实例
# showVideo: 是否显示视频
# saveVideo: 保存视频的writer对象,None不保存
# def frameProcessByDNN(currentframe, hand_model = None, showVideo = True, saveVideo = None):
#     global handTrackLen  # 跟踪的手部运动轨迹长度
#     global handTrack  # 记录手部轨迹坐标元组的循环列表
#     global hPoint  # handTrack列表当前位置指针
#     global conHandTrackLen  # 用连续conHandTrackLen次轨迹判定结果生成最终轨迹
#     global conHandTrack  # 记录手部轨迹识别结果，循环列表
#     global tPoint  # conHandTrack列表当前位置指针
#     t = time.time()
#
#     # 识别手形handShape，置信度ch, 手部位置handPosition, 返回处理后的图像帧
#     handShape, ch, handPosition, colorhand = handShapeRecogbyDNN(currentframe, hand_model)
#     # 用循环列表计算手部轨迹
#     handTrack[hPoint] = handPosition
#     hPoint = (hPoint + 1) % handTrackLen
#     # 识别手部运动handMovement，置信度cm
#     handMovement, cm = handMovRecog(handTrack[hPoint + 1:handTrackLen] + handTrack[0:hPoint + 1], hand=None)
#     conHandTrack[tPoint] = handMovement
#     tPoint = (tPoint + 1) % conHandTrackLen
#     if handMovement not in ('static', 'unkown') and \
#             conHandTrack[conHandTrackLen-1 if (tPoint - 1 == -1) else (tPoint - 1)] not in ('static', 'unkown'):
#         handMovement = max(conHandTrack, key=conHandTrack.count)
#     # 识别手势gesture，置信度cg
#     gesture, cg = gestureRecog(handShape, handMovement, ch, cm)
#     if gesture == 'invalid': cg = 1
#     cg = np.round(cg, 2)
#     # 计算处理速度（平均每帧处理速度）
#     v = np.round((time.time() - t) / (handTrackLen), 5) * 1000
#     v = str(v)[0:4] + ' ms'
#     if showVideo or saveVideo:
#         # 在视频上显示识别结果和性能指标
#         cv2.putText(colorhand, handShape, (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness = 2)
#         cv2.putText(colorhand, handMovement, (150, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness = 2)
#         if gesture == 'invalid':
#             pass
#             # cv2.putText(hand, gesture, (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), thickness=2)
#         else:
#             cv2.putText(colorhand, gesture, (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness = 2)
#         cv2.putText(colorhand, "Time/Frame: " + v, (5, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness = 2)
#         cv2.putText(colorhand, "Confidence: " + str(cg), (400, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness = 2)
#         # 显示视频
#         if showVideo:
#             if VideoWinScale is not None:
#                 showImg(colorhand, 'HandGesture', xScale = VideoWinScale)
#             else:
#                 showImg(colorhand, 'HandGesture', xScale = 1)
#             cv2.waitKey(1)
#         # 保存视频
#         if saveVideo:
#             saveVideo.write(colorhand)
#     return gesture, cg


# 基于深度学习手形识别的帧处理，先做手部分割，分割出的手部用深度学习方法识别
# hand_model：手部识别实例
# def frameProcessByDNNwithHandSeg(currentframe, lastframe, hand_model=None, face_cascade=None, handSize=100,
#                                      useWaterShed=True, moveSeg=True, showVideo=True, saveVideo=None):
#     global handTrackLen  # 跟踪的手部运动轨迹长度
#     global handTrack  # 记录手部轨迹坐标元组的循环列表
#     global hPoint  # handTrack列表当前位置指针
#     global conHandTrackLen  # 用连续conHandTrackLen次轨迹判定结果生成最终轨迹
#     global conHandTrack  # 记录手部轨迹识别结果，循环列表
#     global tPoint  # conHandTrack列表当前位置指针
#     t = time.time()
#
#     originalFrame=currentframe.copy()
#     # 分割出手部轮廓handcontour,手部掩膜hand（二值图像，白色为手部），并算出手部位置handPosition
#     # 如果handPosition=(-1,-1)，未分割到手势
#     handContour, hand, handPosition = segHand(lastframe, currentframe, face_cascade, handSize, useWaterShed, moveSeg)
#
#     height,width,_=currentframe.shape
#     x,y,w,h=cv2.boundingRect(handContour)
#     # 放大矩形框
#     margin=50
#     x1=max(x-margin,0)
#     x2=min(width,x+w+margin)
#     y1=max(y-margin,0)
#     y2=min(height,y+h+margin)
#
#     # cv2.drawContours(originalFrame,[handContour],-1,(0,255,0),-1)
#     # cv2.rectangle(originalFrame,(x,y),(x+w,y+h),(0,255,0),1)
#     handFrame=originalFrame[y1:y2,x1:x2,:]
#     # cv2.imshow('test',handFrame)
#     # cv2.waitKey(1)
#
#     # img_crop will the cropped rectangle, img_rot is the rotated image
#
#     # 识别手形handShape，置信度ch, 手部位置handPosition, 返回处理后的图像帧
#     handShape, ch, handPositionDNN, colorhand = handShapeRecogbyDNN(handFrame, hand_model)
#     originalFrame[y1:y2, x1:x2, :]=colorhand
#     colorhand = originalFrame
#     # handShape, ch, handPosition, colorhand = handShapeRecogbyDNN(originalFrame, hand_model)
#     print('手部识别时间：',time.time()-t)
#     # 用循环列表计算手部轨迹
#     handTrack[hPoint] = handPosition
#     hPoint = (hPoint + 1) % handTrackLen
#     # 识别手部运动handMovement，置信度cm
#     handMovement, cm = handMovRecog(handTrack[hPoint + 1:handTrackLen] + handTrack[0:hPoint + 1], hand=None)
#     conHandTrack[tPoint] = handMovement
#     tPoint = (tPoint + 1) % conHandTrackLen
#     if handMovement not in ('static', 'unkown') and \
#             conHandTrack[conHandTrackLen-1 if (tPoint - 1 == -1) else (tPoint - 1)] not in ('static', 'unkown'):
#         handMovement = max(conHandTrack, key=conHandTrack.count)
#     # 识别手势gesture，置信度cg
#     gesture, cg = gestureRecog(handShape, handMovement, ch, cm)
#     if gesture == 'invalid': cg = 1
#     cg = np.round(cg, 2)
#     # 计算处理速度（平均每帧处理速度）
#     v = np.round((time.time() - t) / (handTrackLen), 5) * 1000
#     v = str(v)[0:4] + ' ms'
#     print('手形：',handShape,'运动轨迹：',handMovement,"每帧处理时间: ",v)
#     if showVideo or saveVideo:
#         # 在视频上显示识别结果和性能指标
#         cv2.putText(colorhand, handShape, (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness = 2)
#         cv2.putText(colorhand, handMovement, (150, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness = 2)
#         if gesture == 'invalid':
#             pass
#             # cv2.putText(hand, gesture, (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), thickness=2)
#         else:
#             cv2.putText(colorhand, gesture, (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness = 2)
#         cv2.putText(colorhand, "Time/Frame: " + v, (5, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness = 2)
#         cv2.putText(colorhand, "Confidence: " + str(cg), (400, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness = 2)
#         # 显示视频
#         if showVideo:
#             if VideoWinScale is not None:
#                 showImg(colorhand, 'HandGesture', xScale = VideoWinScale)
#             else:
#                 showImg(colorhand, 'HandGesture', xScale = 1)
#             cv2.waitKey(1)
#         # 保存视频
#         if saveVideo:
#             saveVideo.write(colorhand)
#     lastFrame = np.copy(currentframe)
#     return gesture, cg, lastFrame