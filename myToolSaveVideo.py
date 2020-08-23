import cv2
import sys,os
from myConfig import *
from myCommonModules import makeVideoFileName

# 读取摄像头视频，保存为视频文件
def saveVideo(cameraID=0,filename='rawvideo'):
    # 读入视频，提取帧图像
    cap = cv2.VideoCapture(cameraID)

    # 强制视频格式转换，防止YUK格式帧率过低
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    print('FrameWidth:', cap.get(cv2.CAP_PROP_FRAME_WIDTH), ', FrameHeight:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hasFrame, frame = cap.read()
    if hasFrame:
        # 如果文件已经存在，自动重命名
        videofilename=makeVideoFileName(filename=filename)
        # 保存视频文件用
        videoWriter = cv2.VideoWriter(videofilename, fourcc, 30,(frame.shape[1], frame.shape[0]))

        while (cap.isOpened()):
            hasFrame, frame = cap.read()
            if hasFrame:
                showframe=cv2.flip(frame,1)
                cv2.imshow('Capturing & saving video...Press ESC for exit',showframe)
                videoWriter.write(frame)
            # 响应键盘，等1ms，按Esc键退出
            key = cv2.waitKey(1)
            if key == 27:
                videoWriter.release()
                cap.release()
                print('视频保存成功为',videofilename,'，程序已退出。')
                break

if __name__ == '__main__':
    saveVideoPath=RawVideoPath # 从配置文件中读取原始视频记录路径
    if len(sys.argv)>1: # 如果命令行传入记录路径，用命令行传入的
        saveVideoPath=sys.argv[1]
    else:
        print("可以用'python myToolSaveVideo filename'命令行指定要保存的视频文件名，可带路径，不要加扩展名")

    # 采集摄像头视频并保存为视频文件
    saveVideo(cameraID=CameraID,filename=saveVideoPath)