import cv2
import numpy as np

# Kalman滤波
class KalmanFilter():
    def __init__(self, id, point):
        self.id = int(id)
        # 设置卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03

        self.measurement = (0, 0)  # 测量状态
        self.current = (0, 0)  # 当前状态
        self.prediction = (0, 0)  # 预测状态
        self.Correct(point)

    def Correct(self, point):  # 更新，并返回更新后的状态
        self.measurement = point
        self.kalman.correct(np.array([[np.float32(point[0])], [np.float32(point[1])]]))  # 当前状态statePost被更新
        self.current = (int(self.kalman.statePost[0][0]), int(self.kalman.statePost[1][0]))
        return self.current

    def Predict(self):  # 预测，并返回预测值
        self.kalman.predict()  # 当前状态statePost和statePre同时被更新为相同的值
        self.prediction = (int(self.kalman.statePre[0][0]), int(self.kalman.statePre[1][0]))
        return self.prediction