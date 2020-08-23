import functools
import asyncio
import websockets
from myConfig import *
from myGlobalVariables import *
from myCommonModules import *
from myFrameProcessing import frameProcess
from sklearn.externals import joblib

gGESTURE = 'invalid'
gConfidence = 1.00

async def recv_frame(websocket, handsvm, face_cascade ,vidRaw_writer, vid_writer,showVideo
                     , handSize, useWaterShed, moveSeg, saveRawVideo, saveVideo, collectHandData):
    global gGESTURE
    global gConfidence

    lastFrame = None
    async for frame in websocket:
        frame = np.fromstring(frame, dtype = 'uint8')
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        if saveRawVideo:  # 保存原始视频
            vidRaw_writer.write(frame)
        if lastFrame is None:
            lastFrame = np.copy(frame)
        gesture, cg, lastFrame = frameProcess(frame, lastFrame, handsvm = handsvm, face_cascade = face_cascade,
                                              useWaterShed = useWaterShed, moveSeg = moveSeg, handSize = handSize,
                                              showVideo = showVideo, saveVideo = vid_writer,
                                              collectHandData = collectHandData)
        gGESTURE = gesture
        gConfidence = cg
        data = {}
        data["gesname"] = gesture
        data["confidence"] = cg
        gestureJSON = json.dumps(data, cls = MyEncoder)
        if cv2.waitKey(1) == 27: #测试
            if saveRawVideo:  # 保存原始视频
                print('正在保存原始视频...', end = '')
                vidRaw_writer.release()
                print('OK')
            if saveVideo:  # 保存处理过的视频
                print('正在保存处理过的视频...', end = '')
                vid_writer.release()
                print('OK')
            # 保存手形数据，采集训练数据时用
            if collectHandData:
                print('正在保存手形数据...', end = '')
                text_save(handdata, collectHandData + '_data.txt')
                print('OK')
            cv2.destroyAllWindows()
            return
        await websocket.send(gestureJSON)


async def main_logic(websocket, path,  hand_svm_model, face_cascade, vidRaw_writer, vid_writer,
                     showVideo=True, handSize= 100, useWaterShed=True, moveSeg=True, saveRawVideo=None, saveVideo=None, collectHandData=None):

    await recv_frame(websocket, hand_svm_model, face_cascade, vidRaw_writer, vid_writer, showVideo
                     , handSize, useWaterShed, moveSeg, saveRawVideo, saveVideo, collectHandData)


if __name__ == '__main__':

    hand_svm_model = HandShapeSVMPath
    face_cascade_path = FaceCascadePath
    Height = FrameHeight
    Width = FrameWidth
    saveRawVideo = None
    saveVideo = None
    collectHandData = None

    print('加载SVM手形识别模型...', end = '')
    handsvm = None
    if not hand_svm_model is None:
        if os.path.exists(hand_svm_model):
            handsvm = joblib.load(hand_svm_model)  # 加载训练好的手形识别svm模型
    print("OK")
    # 加载Haar人脸检测器
    face_cascade = None
    if not face_cascade_path is None:
        print('加载人脸检测模型...', end = '')
        face_cascade = cv2.CascadeClassifier(face_cascade_path)  # 加载级联分类器模型
        face_cascade.load(face_cascade_path)
        print('OK')

    # 保存视频处理结果用
    vidRaw_writer = None
    vid_writer = None
    if saveRawVideo:
        print('创建原始视频存储对象...', end = '')
        saveRawVideoFileName = makeVideoFileName(filename = saveRawVideo)
        vidRaw_writer = cv2.VideoWriter(saveRawVideoFileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                        (Width, Height))
        print('OK')
    if saveVideo:
        print('创建处理过的视频存储对象...', end = '')
        saveRecVideoFileName = makeVideoFileName(filename = saveVideo)
        vid_writer = cv2.VideoWriter(saveRecVideoFileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                     (Width * 2, Height))
        print('OK')

    # 创建目录，采集手形数据图像用
    if collectHandData:
        if not os.path.exists(collectHandData + '_imgs'):
            print('创建存储手形图像的文件夹' + collectHandData + '_imgs' + '...', end = '')
            os.makedirs(collectHandData + '_imgs')
            print('OK')
    print('启动Socket服务器...', end = '')
    start_server = websockets.serve(functools.partial(main_logic, hand_svm_model = handsvm, face_cascade = face_cascade,
                                                  vidRaw_writer = vidRaw_writer, vid_writer = vid_writer, showVideo = True,
                                                  handSize = HandSize, useWaterShed = UseWaterShed,
                                                  moveSeg = True, saveRawVideo = None, saveVideo = None, collectHandData=None), '0.0.0.0', WebSocketServerPort)
    print('Socket服务器端已启动，开始侦听客户端连接......')
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
