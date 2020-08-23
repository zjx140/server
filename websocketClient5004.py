from myConfig import *
from myWebsockets import client_connect_recv
import asyncio

if __name__ == '__main__':
    print('启动Websock客户端连接5004...')
    asyncio.get_event_loop().run_until_complete(client_connect_recv(ip=VideoDataIP, port=str(VideoDataPort),
                                                                    svm_hand_model_path = HandShapeSVMPath,
                                                                    dnn_hand_model_path=HandShapeDNNPath,
                                                                    face_cascade_path = FaceCascadePath,
                                                                    resFaceModel_path= ResFaceModelPath,
                                                                    useDlibFace=UseDlibFace,
                                                                    handAreaScope= HandAreaScope,
                                                                    useWaterShed=UseWaterShed, moveSeg=UseMoveSeg,
                                                                    useBackSeg=UseBackSeg,saveRawVideo=RawVideoPath,
                                                                    saveVideo=RecVideoPath))

    print('WebSocket5004客户端程序已退出')

    #print('启动Websock客户端连接5002并发送startCapturingVideo指令...')
    #cmd='startCapturingVideo'
    #asyncio.get_event_loop().run_until_complete(client_connect_send(ip='127.0.0.1', port='5002',cmd=cmd))
    #print('WebSocket5002 Start客户端程序已退出')