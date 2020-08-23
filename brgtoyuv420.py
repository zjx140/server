import cv2
import numpy as np

img = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
print(img.shape)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
print(img_yuv.shape)
cv2.imshow("yuv", img_yuv)
img_change = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_I420)
print(img_change.shape)
cv2.imshow("orginal",img)
cv2.imshow("change",img_change)
cv2.waitKey(0)
cv2.destroyAllWindows()
