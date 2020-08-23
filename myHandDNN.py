from __future__ import division
from myConfig import CameraID
import cv2
import time
import tensorflow as tf
import json
import os
from myCommonModules import inRect
import numpy as np

# 手部深度模型
class handDNN(object):
    # modelpath为手部Caffe模型所在路径
    def __init__(self, model_path,confidence=0.8, ratio = 0.15):
        modelpath = os.path.join(model_path, r"frozen_inference_graph.pb")
        labelpath = os.path.join(model_path, r'labels.json')
        self.sess = self.loadHandModel(modelpath) # 加载手部模型
        self.label = json.load(open(labelpath, 'r'))
        self.ratio = ratio
        self.confidence=confidence # 置信度阈值

    # 加载手部关键点Caffe模型
    # 输入：modelpath为模型所在路径
    def loadHandModel(self, modelpath):
        config = tf.compat.v1.ConfigProto(
            gpu_options = tf.compat.v1.GPUOptions(allow_growth = True))

        with tf.compat.v1.Graph().as_default() as net_graph:
            graph_data = tf.io.gfile.GFile(modelpath, 'rb').read()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_data)
            tf.import_graph_def(graph_def, name = '')
            # summaryWriter = tf.compat.v1.summary.FileWriter('log/', net_graph)

        sess = tf.compat.v1.Session(graph = net_graph, config = config)
        sess.graph.as_default()

        return sess

    # faces传入人脸框[(x, y, w, h),(x, y, w, h),...]，辅助判断检测到手部位置的正确性
    def predict(self, frame):
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        larger_factor = np.array([[frame_height, frame_width, frame_height, frame_width]])
        inp = cv2.resize(frame, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        out = self.sess.run(
            [
                self.sess.graph.get_tensor_by_name('num_detections:0'),
                self.sess.graph.get_tensor_by_name('detection_scores:0'),
                self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                self.sess.graph.get_tensor_by_name('detection_classes:0')
            ],
            feed_dict = {
                'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)
            },
        )

        class_name=None
        bbox=None
        score=1.0
        beyond_stan_score_args = np.where(out[1][0] >= self.confidence)
        scores = out[1][0][beyond_stan_score_args]
        bboxs = out[2][0][beyond_stan_score_args]
        class_ids = out[3][0][beyond_stan_score_args].copy().astype(np.int8)
        if len(scores) >= 1:
            #得到真实的点的位置
            bboxs_real_len = larger_factor * bboxs
            #获取所有的边框面积取最小
            bboxs_areas = np.abs((bboxs_real_len[:,2] - bboxs_real_len[:,0]) * (bboxs_real_len[:,3] - bboxs_real_len[:,1]))
            bboxs_areas_min_arg = np.argmin(bboxs_areas)
            #print(scores[bboxs_areas_min_arg])
            #计算占比
            ratio = np.min(bboxs_areas) / (frame_height * frame_height)
            if ratio <= self.ratio:
                score = float(scores[bboxs_areas_min_arg])
                y1, x1, y2, x2 = int(bboxs_real_len[bboxs_areas_min_arg][0]),int( bboxs_real_len[bboxs_areas_min_arg][1]), \
                                 int(bboxs_real_len[bboxs_areas_min_arg][2]),int( bboxs_real_len[bboxs_areas_min_arg][3])
                class_id = class_ids[bboxs_areas_min_arg]
                class_name = self.label[class_id - 1]
                bbox = y1, x1, y2, x2
        return class_name, bbox, score

        # num_detections = int(out[0][0])
        # for i in range(num_detections):#score在out里从大到小排，取最大的和其对应的识别标签和位置
        #     score = float(out[1][0][i])
        #     if score>=self.confidence:# 置信度高于阈值
        #         bbox = [float(v) for v in out[2][0][i]]
        #         class_id = int(out[3][0][i])
        #         # 位置坐标换算
        #         x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
        #         x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)
        #         handCenter = (int((x1+x2)/2),int((y1+y2)/2))
        #         isFace=False
        #         for face in faces:
        #             if inRect(face,handCenter):# 手部是否落在人脸框中
        #                 isFace=True
        #                 break
        #             else:
        #                 handRect=[[x1,y1,x2-x1,y2-y1]]
        #                 faceCenter=(int(face[0]+face[2]/2),int(face[1]+face[3]/2))
        #                 if inRect(handRect,faceCenter): # 手部框是否把人脸中心包含进来
        #                     isFace = True
        #                     break
        #         if isFace:
        #             continue
        #         else:
        #             bbox = (y1, x1, y2, x2)
        #             #从标号找标签名
        #             class_name = self.label[class_id - 1]
        #
        # return class_name, bbox, score


if __name__ == '__main__':
    print("启动手势识别...")
    start = time.time()
    modelpath = r"../model/"
    myHand = handDNN(modelpath,confidence = 0.7)
    print("模型加载时间: ", time.time() - start)

    cap = cv2.VideoCapture(CameraID)
    while True:
        start = time.time()
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        if not ret:
            break
        class_name, bbox, score = myHand.predict(frame)
        print("单帧检测识别时间: ", time.time() - start)
        if bbox is None:
            cv2.imshow('HandRecognition', frame)
        else:
            y1, x1, y2, x2 = bbox
            cv2.putText(frame, class_name + ":" + '{:.3f}'.format(score),
                    (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('HandRecognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

