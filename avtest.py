import av
from io import BytesIO
import cv2

f = open('D:\\test.264','rb')
byte_str = f.read()
s = BytesIO(byte_str)
container = av.open(s)
index = 0
for frame in container.decode(video = 0):
    index += 1
    img = frame.to_nd_array(format = 'bgr24')
    cv2.imshow('Client', img)
    cv2.imwrite("Test{}.jpg".format(index), img)