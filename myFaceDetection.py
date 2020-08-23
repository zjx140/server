import cv2
import os
import dlib


class faceDection(object):
    # 初始化，加载人脸检测模型
    # useDlibFace=True,使用Dlib Hog人脸检测器
    # face_cascade_path=None,Haar人脸检测模型路径
    # resFaceModel_path=None，Resnet人脸检测模型路径
    def __init__(self,useDlibFace=True,face_cascade_path=None,resFaceModel_path=None):
        # 人脸检测器均初始化为空值
        self.dlibFace = None
        self.face_cascade = None
        self.resNetFace = None

        # 加载Dlib Hog人脸检测器，优先级最高，速度快，较准确
        if useDlibFace:
            print('加载Dlib Hog人脸检测器')
            self.dlibFace = dlib.get_frontal_face_detector()
        elif not resFaceModel_path is None:# 加载Resnet人脸检测模型，优先级次之，最稳定，速度慢
                print('加载Resnet人脸检测器')
                prototxt =os.path.join(resFaceModel_path, "res10_deploy.prototxt")
                caffemodel = os.path.join(resFaceModel_path, "res10_300x300_ssd_iter_140000.caffemodel")
                self.resNetFace = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        elif not face_cascade_path is None:# 加载Haar人脸检测器，优先级最低，速度最快，最不稳定
            print('加载Haar人脸检测器')
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)  # 加载级联分类器模型
            self.face_cascade.load(face_cascade_path)

    def detection(self,currentFrame):
        self.faces=[]
        if not self.dlibFace is None:# 使用Dlib Hog人脸检测，较快，较准，优先使用
            rgbImg=cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB) # Dlib要求RGB图像
            dets = self.dlibFace(rgbImg, 0)  # 使用Dlib进行人脸检测 dets为返回的结果
            for index, face in enumerate(dets):
                self.faces.append([face.left(), face.top(), int(face.right() - face.left()), int(face.bottom() - face.top())])
        elif not self.resNetFace is None:# 利用ResNet模型检测人脸
            cols = currentFrame.shape[1]
            rows = currentFrame.shape[0]
            inWidth, inHeight=300,300
            confThreshold = 0.5 # 置信度阈值
            self.resNetFace.setInput(cv2.dnn.blobFromImage(currentFrame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
            detections = self.resNetFace.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confThreshold:
                    xLeftBottom = int(detections[0, 0, i, 3] * cols)
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop = int(detections[0, 0, i, 5] * cols)
                    yRightTop = int(detections[0, 0, i, 6] * rows)
                    self.faces.append([xLeftBottom,yLeftBottom,xRightTop-xLeftBottom,yRightTop-yLeftBottom])
                    # cv2.rectangle(currentFrame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))
            # self.perf_stats = self.resNetFace.getPerfProfile()
            # print('Inference time, ms: %.2f' % (self.perf_stats[0] / cv2.getTickFrequency() * 1000))
        elif not self.face_cascade is None:# 利用Haar特征检测人脸
            grayFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
            # grayFrame= cv2.equalizeHist(grayFrame) # 亮度均衡，并不会改善人脸检测效果
            self.faces = self.face_cascade.detectMultiScale(grayFrame, 1.2, 4, cv2.CASCADE_SCALE_IMAGE, (100, 100))
        return self.faces

    # 画人脸框（绿色框，边框宽度为2）
    def visFaces(self,img,color=(0, 255, 0),thick=2):
        for (x, y, w, h) in self.faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        return img

if __name__ == '__main__':
    myFace=faceDection()
    cap=cv2.VideoCapture(0)
    while True:
        hasframe,img=cap.read()
        img=cv2.flip(img,1)
        faces=myFace.detection(img)
        myFace.visFaces(img)
        cv2.imshow('Face',img)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()