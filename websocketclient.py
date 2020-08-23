import cv2
import websockets
import asyncio #调用python异步IO库
import numpy as np
from myConfig import *
from myMapDemo import baiduMap
import json


async def send_frame(websocket, cap, nFrames = 2, showVideo = True, DemoViaSocket = True, sensitivity = 1):
    if DemoViaSocket:
        mapDemo = baiduMap(mapPath=MapPath, mapType='BMAP_NORMAL_MAP', chromeDriverPath=ChromeDriverPath, sensitivity= sensitivity)
        DemoViaSocket = mapDemo.canDemo
    k = 0
    sendBUFSIZE = FrameWidth * FrameHeight * 3
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            cv2.imshow("Client", frame)

            k = (k + 1) % 2592000
            if k % nFrames == 0:
                img_encode = cv2.imencode('.jpg', frame)[1]
                rows, _ = img_encode.shape
                data = np.array(img_encode)
                len0, _ = data.shape
                # 加0补位，不加0会产#生TCP粘包问题,其中BUFFER_SIZE是TCP两端约定的数值，必须相同
                data0 = np.zeros((sendBUFSIZE - len0, 1), dtype = np.uint8)
                finaldata = np.vstack((data, data0))  # 合并
                stringdata = finaldata.tostring()
                await websocket.send(stringdata) #异步发送
                try:
                    recvData = await websocket.recv()
                except websockets.ConnectionClosedOK as e:
                    print("服务端已关闭，客户端退出")
                    await websocket.close()
                    return False
                gescfdata = json.loads(recvData)
                #recvData = recvData.decode('utf-8')
                if DemoViaSocket:
                    try:
                        mapDemo.run(command = gescfdata['gesname'])
                        if mapDemo.confirmClose:
                            break
                    except Exception as e:
                        print('执行动作出错：',e)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            await websocket.close(reason = 'user exit')
            return False


async def main_logic(cameraID = 0, ip = '127.0.0.1', port = 7777, nFrames = 2, showVideo=True, DemoViaSocket=True, sensitivity=1):
    ws = 'ws://' + ip + ':' + str(port)
    async with websockets.connect(ws) as websocket:
        cap = cv2.VideoCapture(cameraID, cv2.CAP_DSHOW)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        await send_frame(websocket, cap, nFrames, showVideo, DemoViaSocket, sensitivity)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main_logic(cameraID = CameraID, ip = WebSocketServerIP, port = WebSocketServerPort, nFrames = ProcessOneEverynFrames,
                                                       showVideo = True, DemoViaSocket = DemoMap, sensitivity = Sensitivity))

